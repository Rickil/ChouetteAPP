from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import String, Date
from sqlalchemy.orm import Mapped, mapped_column
from datetime import date
from sqlalchemy import create_engine
import os

#Models creation
Base = declarative_base()

class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(30))
    weights_path: Mapped[str] = mapped_column(String)
    dataset_start_date: Mapped[date] = mapped_column(Date)
    dataset_end_date: Mapped[date] = mapped_column(Date)

    def __repr__(self):
        return f"({self.model_name}, {self.dataset_start_date}:{self.dataset_end_date}), path: {self.weights_path}"

#Engine creation
engine = create_engine(os.getenv("DB_URL"), echo=False)

#Database intitialisation
Base.metadata.create_all(engine)