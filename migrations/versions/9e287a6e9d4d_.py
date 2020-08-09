"""empty message

Revision ID: 9e287a6e9d4d
Revises: 
Create Date: 2020-08-08 23:46:34.117322

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9e287a6e9d4d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('agents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('process_name', sa.Text(), nullable=True),
    sa.Column('agent_name', sa.Text(), nullable=True),
    sa.Column('options', sa.PickleType(), nullable=True),
    sa.Column('data', sa.PickleType(), nullable=True),
    sa.Column('features', sa.PickleType(), nullable=True),
    sa.Column('model', sa.PickleType(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('author', sa.String(length=64), nullable=True),
    sa.Column('scope', sa.String(length=64), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agents_timestamp'), 'agents', ['timestamp'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=64), nullable=True),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('role', sa.Integer(), nullable=True),
    sa.Column('password_hash', sa.String(length=128), nullable=True),
    sa.Column('org', sa.String(length=64), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_agents_timestamp'), table_name='agents')
    op.drop_table('agents')
    # ### end Alembic commands ###
