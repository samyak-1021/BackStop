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
    # None for an irreversible effect (a sent notification). Keeping these in
    # the log anyway is deliberate: the report needs to know they happened.
    compensate: Callable[[], Awaitable[None]] | None
    detail: dict | None = None

    @property
    def reversible(self) -> bool:
        return self.compensate is not None


@dataclass
class CompensationResult:
    """What happened when we tried to unwind."""

    compensated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    irreversible: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when every reversible effect was successfully undone.

        Irreversible effects do not make an unwind dirty *here* — the runtime's
        job was to avoid performing them, and whether it managed that is the
        verifier's call, not the saga's.
        """
        return not self.failed


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
    ) -> None:
        """Register a completed side effect and how to undo it."""
        self._effects.append(Effect(name=name, compensate=compensate, detail=detail))

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
        """
        return any(not e.reversible for e in self._effects)

    async def unwind(self, attempts: int = 3) -> CompensationResult:
        """Undo every reversible effect, newest first.

        Each compensation gets its own retries: the tools that undo things sit
        behind the same faulty network as the tools that did them, so a
        single-shot unwind would leave orphans exactly when the world is at its
        worst — which is the moment unwinding matters most.
        """
        result = CompensationResult()

        for effect in reversed(self._effects):
            if effect.compensate is None:
                result.irreversible.append(effect.name)
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
                result.failed.append((effect.name, last_error))

        return result
