# MItra Query Hub

A Streamlit chat app that lets non-technical program teams ask natural-language questions
about MItra's grassroots data (Story and Discussion bot submissions from Bihar/Karnataka)
and get answers generated via a text-to-SQL LangChain pipeline.

## Architecture

- **`analytics_db`** — the one analytical database the app queries (Postgres). Contains
  MItra's `submissions`, `story_submissions`, `discussion_submissions`, `themes`, `programs`,
  and related tables.
- **`query_hub` (Postgres)** — a separate database, used only to persist chat sessions,
  logged queries/responses, and 👍/👎 feedback. Not the data being queried — just the app's
  own operational data, kept apart from `analytics_db` so the app never writes into the
  analytics database itself. Configured fully independently from `analytics_db` (own
  host/port/credentials) — it can live on the same Postgres server or a completely
  different one.
- LLM calls go through **OpenRouter** (`langchain_openai.ChatOpenAI` pointed at OpenRouter's
  base URL), so the model can be swapped via env vars without code changes.

```
app/            Streamlit UI (entrypoint: app/main.py)
core/           LLM client, retrieval + interpretation chains, context/history
db/             App DB (Postgres): models, engine, repositories
data/           table_desc_analytics_db.csv - schema descriptions fed to the LLM
config/         env-driven settings (DB connection strings, model names)
prompts.py      the MItra system/answer prompt
```

## Setup

1. Create a virtualenv and install dependencies:
   ```bash
   python3 -m venv myenv && source myenv/bin/activate
   pip install -r requirements.txt
   ```
2. Create the app's own database (once), wherever it will live:
   ```bash
   createdb -h $APP_DB_HOST -p $APP_DB_PORT -U $APP_DB_USER query_hub
   ```
3. Copy `.env.example` to `.env` and fill in:
   - `ANALYTICS_DB_USER`/`PASSWORD`/`HOST`/`PORT`/`NAME` — connection to `analytics_db`.
   - `APP_DB_USER`/`PASSWORD`/`HOST`/`PORT`/`NAME` — connection to the app's own database
     created in step 2 (default name `query_hub`). Independent of the analytics connection —
     can point to a different host/server entirely.
   - `OPENROUTER_API_KEY` — required; get one from [openrouter.ai](https://openrouter.ai).
   - `RETRIEVAL_MODEL` / `INTERPRETATION_MODEL` — OpenRouter model ids (default:
     `anthropic/claude-sonnet-5` for both). Change these to try a different model — no
     code change needed.
4. Run the app:
   ```bash
   streamlit run app/main.py
   ```
   On first run this also creates the `sessions`/`queries`/`feedback` tables in the
   `query_hub` database, via `db.engine.init_db()`.

## Notes

- **Local-machine assumption**: the default `analytics_db` connection (`localhost`, trust
  auth) only works when Streamlit and Postgres run on the same machine. A future deployment
  to a separate host will need real credentials and a `pg_hba.conf` policy change — don't
  let the current defaults become the assumed production setup.
- **Session persistence**: each browser tab gets a `session_id` stored in the URL query
  string (`?session_id=...`) so refreshing the page reloads that session's chat history
  from the `query_hub` database instead of starting over.
- **Updating `data/table_desc_analytics_db.csv`**: if `analytics_db`'s schema changes,
  update this file — it's the only thing telling the LLM what tables exist and what they're
  for.
