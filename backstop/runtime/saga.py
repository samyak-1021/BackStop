"""The compensation log.

A multi-step task with side effects is a saga: a sequence of local commits, each
with a compensating action that semantically undoes it. There is no distributed
transaction to roll back — you cannot un-charge a card with a ``ROLLBACK`` — so
correctness comes from *deliberately* undoing what you did, in reverse.

Three properties this implementation takes seriously:

**Record before you can forget.** An effect is registered the moment it is known
to have happened, including when it was discovered after a lost response. An
effect that isn't in the log will never be compensated.

**Unwind in reverse.** Later steps depend on earlier ones (a shipment consumes a
reservation), so compensations must run newest-first or they conflict.

**Compensation is best-effort, and its failures are reported.** The tools that
undo things can fail too. A compensation that fails leaves a real orphan, so it
is retried and, if it still fails, recorded — never silently swallowed. A
runtime that claims a clean unwind it didn't achieve is worse than one that
admits the mess.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class Effect:
    """Something that happened in the world, and how to undo it."""

    name: str
    # None when we hold no way to undo this effect right now. That covers two
    # very different situations, which is why `irreversible` is separate.
    compensate: Callable[[], Awaitable[None]] | None
    detail: dict | None = None
    # True only when the effect genuinely cannot be undone by anyone — a sent
    # email. NOT true merely because we lack a handle for it.
    #
    # Conflating these two was a real bug: a capture whose response was lost is
    # perfectly refundable, we just don't know its id yet, and reconciliation
    # will find it. Treating that uncertainty as irreversibility tripped the
    # point-of-no-return rule, suppressed the unwind, and left money taken with
    # nothing shipped — the exact harm the unwind existed to prevent.
    irreversible: bool = False

    @property
    def has_compensation(self) -> bool:
        return self.compensate is not None


@dataclass
class CompensationResult:
    """What happened when we tried to unwind."""

    compensated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    irreversible: list[str] = field(default_factory=list)
    # The effect the unwind stopped at, if it could not run to completion.
    halted_at: str | None = None

    @property
    def clean(self) -> bool:
        """True when the unwind actually restored the world.

        This used to be ``not self.failed``, which let the saga report a clean
        unwind while leaving an effect standing — the exact thing this module's
        docstring says is worse than admitting the mess. An unwind that stopped
        early did not achieve anything of the sort.
        """
        return not self.failed and self.halted_at is None


class SagaLog:
    """An append-only record of effects, unwound in reverse on failure."""

    def __init__(self) -> None:
        self._effects: list[Effect] = []

    def __len__(self) -> int:
        return len(self._effects)

    @property
    def effects(self) -> list[Effect]:
        return list(self._effects)

    def record(
        self,
        name: str,
        compensate: Callable[[], Awaitable[None]] | None = None,
        detail: dict | None = None,
        irreversible: bool = False,
    ) -> None:
        """Register a completed side effect and how to undo it."""
        self._effects.append(
            Effect(
                name=name,
                compensate=compensate,
                detail=detail,
                irreversible=irreversible,
            )
        )

    def has(self, name: str) -> bool:
        return any(e.name == name for e in self._effects)

    @property
    def past_point_of_no_return(self) -> bool:
        """True once an irreversible effect has been performed.

        Beyond this point rolling back is *destructive*, not corrective. If the
        customer has already been told their order shipped, cancelling the
        shipment does not restore the world — it converts a recoverable
        situation into a permanent lie. The only safe moves left are to roll
        forward or to escalate to a human.

        Found the hard way: an earlier version unwound unconditionally, and at
        a 60% fault rate it manufactured 18 false notifications per 200
        episodes out of runs where the goods had genuinely shipped and only the
        final bookkeeping call had failed.

        Reads ``irreversible``, not "has no compensation". Not knowing how to
        undo something is a gap to be closed by reconciliation; being unable to
        undo it is a fact about the world.
        """
        return any(e.irreversible for e in self._effects)

    async def unwind(
        self, attempts: int = 3, halt_at_uncompensatable: bool = True
    ) -> CompensationResult:
        """Undo every reversible effect, newest first.

        Each compensation gets its own retries: the tools that undo things sit
        behind the same faulty network as the tools that did them, so a
        single-shot unwind would leave orphans exactly when the world is at its
        worst — which is the moment unwinding matters most.
        """
        result = CompensationResult()

        for effect in reversed(self._effects):
            if effect.compensate is None:
                # Nothing to call: either genuinely irreversible, or an effect
                # whose handle we never learned. **Stop here.**
                #
                # Skipping it and carrying on with the earlier effects is the
                # bug that produced this project's most-quoted finding. The log
                # is ordered by dependency — the payment was captured *for* the
                # shipment — so refunding a capture whose shipment we could not
                # cancel converts "money taken, goods shipped", which is
                # consistent, into `shipped_without_payment`, which is an
                # orphan that did not exist before the unwind ran.
                #
                # This is the point-of-no-return rule one level down. There it
                # is "do not unwind past something irreversible"; here it is "do
                # not unwind past something you could not undo". Same reason:
                # an unwind is only safe as a *complete suffix* of the log.
                result.irreversible.append(effect.name)
                if halt_at_uncompensatable:
                    result.halted_at = effect.name
                    break
                continue

            last_error = ""
            for _ in range(attempts):
                try:
                    await effect.compensate()
                    result.compensated.append(effect.name)
                    break
                except Exception as exc:
                    last_error = str(exc)
            else:
                # A compensation that exhausted its retries leaves that effect
                # standing, so everything it depends on has to stay too.
                result.failed.append((effect.name, last_error))
                if halt_at_uncompensatable:
                    result.halted_at = effect.name
                    break

        return result
