"""Version linear forecasts and allow unavailable confidence; preserves old rows."""
from alembic import op
import sqlalchemy as sa
revision = 'd7e9a142bb01'
down_revision = 'a10a98a6dd49'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('prediction', sa.Column('model_version',sa.String(80),nullable=True))
    op.add_column('prediction', sa.Column('input_level_captured_at',sa.DateTime(timezone=True),nullable=True))
    op.add_column('prediction', sa.Column('generated_at',sa.DateTime(timezone=True),nullable=True))
    op.alter_column('prediction','confidence_score',existing_type=sa.Float(),nullable=True)


def downgrade():
    # Keeping nullable confidence preserves linear rows without inventing a score.
    op.drop_column('prediction','generated_at')
    op.drop_column('prediction','input_level_captured_at')
    op.drop_column('prediction','model_version')
