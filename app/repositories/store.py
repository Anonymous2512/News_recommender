"""Durable store (SQLAlchemy; SQLite by default, any SQLAlchemy URL such as PostgreSQL works).

Holds article metadata, seeded user histories (latest MIND history per user from the artifact) and
recorded click events. Embedding vectors/indexes stay in the artifact files, not in SQL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"
    news_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(1024), default="")
    category: Mapped[str] = mapped_column(String(64), default="", index=True)
    subcategory: Mapped[str] = mapped_column(String(128), default="")


class HistorySeed(Base):
    __tablename__ = "user_history_seed"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    news_id: Mapped[str] = mapped_column(String(32))


class ClickEvent(Base):
    __tablename__ = "click_events"
    __table_args__ = (UniqueConstraint("event_id", name="uq_click_event_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(128))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    news_id: Mapped[str] = mapped_column(String(32), ForeignKey("articles.news_id"))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Meta(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(1024))


@dataclass(frozen=True)
class UserHistory:
    user_id: str
    news_ids: list[str]  # oldest -> newest (seed history, then recorded clicks in arrival order)
    version: str  # changes whenever a click is recorded
    n_seed: int
    n_clicks: int

    @property
    def is_known(self) -> bool:
        return self.n_seed + self.n_clicks > 0


@dataclass(frozen=True)
class ClickResult:
    created: bool
    history_version: str


def make_engine(database_url: str):
    if database_url.startswith("sqlite:///"):
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, future=True,
                           connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {})
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _):  # pragma: no cover - trivial
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return engine


class Repository:
    def __init__(self, database_url: str):
        self.engine = make_engine(database_url)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def close(self) -> None:
        self.engine.dispose()

    # ---------------------------------------------------------------- catalogue
    def get_article(self, news_id: str) -> dict | None:
        with self._session() as s:
            a = s.get(Article, news_id)
            return None if a is None else _article_dict(a)

    def get_articles(self, news_ids: list[str]) -> dict[str, dict]:
        if not news_ids:
            return {}
        with self._session() as s:
            rows = s.scalars(select(Article).where(Article.news_id.in_(news_ids))).all()
            return {a.news_id: _article_dict(a) for a in rows}

    def article_count(self) -> int:
        with self._session() as s:
            return int(s.scalar(select(func.count()).select_from(Article)) or 0)

    def categories(self) -> set[str]:
        with self._session() as s:
            return set(s.scalars(select(Article.category).distinct()).all())

    def meta(self, key: str) -> str | None:
        with self._session() as s:
            m = s.get(Meta, key)
            return None if m is None else m.value

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return True

    # ---------------------------------------------------------------- history / events
    def get_user_history(self, user_id: str, max_items: int = 200) -> UserHistory:
        with self._session() as s:
            seed = s.scalars(select(HistorySeed.news_id).where(HistorySeed.user_id == user_id)
                             .order_by(HistorySeed.position)).all()
            clicks = s.execute(select(ClickEvent.id, ClickEvent.news_id).where(ClickEvent.user_id == user_id)
                               .order_by(ClickEvent.id)).all()
        ids = list(seed) + [c.news_id for c in clicks]
        last_click = clicks[-1].id if clicks else 0
        return UserHistory(user_id=user_id, news_ids=ids[-max_items:], version=f"s{len(seed)}-c{last_click}",
                           n_seed=len(seed), n_clicks=len(clicks))

    def record_click(self, event_id: str, user_id: str, news_id: str, occurred_at: datetime | None) -> ClickResult:
        """Idempotent on ``event_id``. Commits before returning (cache invalidation happens after)."""
        now = datetime.now(timezone.utc)
        with self._session() as s:
            existing = s.scalar(select(ClickEvent).where(ClickEvent.event_id == event_id))
            if existing is not None:
                if existing.user_id != user_id or existing.news_id != news_id:
                    raise ValueError("event_id was already used for a different click")
                created = False
            else:
                s.add(ClickEvent(event_id=event_id, user_id=user_id, news_id=news_id,
                                 occurred_at=occurred_at or now, recorded_at=now))
                try:
                    s.commit()
                    created = True
                except IntegrityError:  # concurrent duplicate
                    s.rollback()
                    created = False
        return ClickResult(created=created, history_version=self.get_user_history(user_id, 1).version)


def _article_dict(a: Article) -> dict:
    return {"news_id": a.news_id, "title": a.title, "category": a.category, "subcategory": a.subcategory}


def import_catalog(database_url: str, artifact_dir: str | Path) -> dict:
    """(Re)load article metadata and seed histories from an artifact. Recorded click events are kept."""
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    articles = pd.read_parquet(artifact_dir / "articles.parquet").fillna("")
    hist_path = artifact_dir / "user_histories.parquet"
    histories = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame(columns=["user_id", "history"])
    repo = Repository(database_url)
    try:
        with Session(repo.engine) as s, s.begin():
            existing = set(s.scalars(select(Article.news_id)).all())
            rows = [{"news_id": r.news_id, "title": r.title[:1024], "category": r.category,
                     "subcategory": r.subcategory} for r in articles.itertuples(index=False)]
            new_rows = [r for r in rows if r["news_id"] not in existing]
            old_rows = [r for r in rows if r["news_id"] in existing]
            if new_rows:
                s.execute(insert(Article), new_rows)
            if old_rows:
                s.execute(update(Article), old_rows)  # bulk UPDATE by primary key
            s.execute(delete(HistorySeed))
            known = set(articles["news_id"])
            s.bulk_save_objects([
                HistorySeed(user_id=r.user_id, position=i, news_id=n)
                for r in histories.itertuples(index=False)
                for i, n in enumerate(n for n in list(r.history) if n in known)
            ])
            digest = hashlib.sha256("".join(sorted(articles["news_id"])).encode()).hexdigest()[:12]
            catalog_version = f"{manifest['artifact_version']}-{digest}"
            for k, v in {"catalog_version": catalog_version, "artifact_version": manifest["artifact_version"],
                         "data_source": manifest["data_source"]}.items():
                s.merge(Meta(key=k, value=v))
        return {"articles": int(len(articles)), "users_seeded": int(len(histories)),
                "catalog_version": catalog_version}
    finally:
        repo.close()
