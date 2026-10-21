"""Remove the unused Support role (#757).

VoxBento has four roles (Admin, Event owner, Room coordinator, Interpreter).
Support was only ever assigned automatically to a third-party app's developer,
and that assignment failed, so this normally deletes nothing. Any rows that do
exist are removed rather than mapped to another role, because Support granted
no more than viewing booths.

The downgrade does not restore deleted rows.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

TABLES = ("event_memberships", "room_memberships", "booth_memberships", "invite_tokens")


def upgrade() -> None:
    conn = op.get_bind()
    existing = set(sa.inspect(conn).get_table_names())
    for table in TABLES:
        if table not in existing:
            continue
        result = conn.execute(sa.text(f"DELETE FROM {table} WHERE role = 'support'"))  # noqa: S608 - fixed table names
        if result.rowcount:
            logger.info("Removed %d Support role rows from %s", result.rowcount, table)


def downgrade() -> None:
    pass
