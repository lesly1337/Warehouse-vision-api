from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime
import datetime

DATABASE_URL = "postgresql+asyncpg://admin:secretpassword@db:5432/warehouse_stats"

engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()
async_sessionmaker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class ProcessingHistory(Base):
    __tablename__ = "processing_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    total_boxes = Column(Integer)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)