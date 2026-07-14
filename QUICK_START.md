# Quick Start Guide

Use these steps to run the ThinHarness-based PineScript Expert agent. See [README.md](README.md) and [scripts.md](scripts.md) for more detail.

## 1. Set Up Environment

Make sure you have Python 3.11 or later installed, then:

```bash
# Install thinharness>=0.5.3 and the RAG dependencies
pip install -r requirements.txt

# Create .env file with API keys
cp .env.example .env
# Edit .env with your API keys
```

## 2. Set Up Database

```bash
# Start PostgreSQL with pgvector using Docker
docker run --name pinescript-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 54322:5432 \
  -v ~/pinescript_postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16

# Initialize the database
python init_db.py
```

## 3. Crawl Documentation & Populate Database

```bash
# Run the crawler to get PineScript documentation
python pinescript_crawler.py
```

## 4. Use the Agent

```bash
# Run interactive mode
python run.py interactive

# Or run a single query
python run.py query "How do I create a moving average in Pine Script?"
```

The interactive shell remembers the full conversation. Enter `clear` to reset that context. Each answer is bounded to 8 model requests and 8 tool calls.

ThinHarness writes prompts, model output, and tool payloads as JSON Lines files in `~/.thinharness/traces/` by default. Disable local tracing with:

```bash
export THINHARNESS_DISABLE_LOCAL_TRACING=1
```

Code that builds a harness can instead set `HarnessConfig(local_tracing=False)`. The Streamlit UI also saves `chat_resume.json` beside `chat_history.pkl`; these files may contain the full transcript and provider reasoning data, so treat them as sensitive.

## Checking Database Status

```bash
# Check database setup
python db_inspect.py count

# View sample documents
python db_inspect.py list

# Test search functionality
python db_inspect.py search "moving average crossover"
```

## Troubleshooting

1. **Docker Issues**: Make sure Docker is running and port 54322 is available
2. **Database Connection**: Verify connection string in `.env` file
3. **Missing pgvector**: Run `python init_db.py` to check if pgvector is correctly installed
4. **OpenAI or model issues**: Run `python api_debug.py`. It checks the OpenAI key and client, then makes a small ThinHarness model request. This command uses a live model and may incur a charge.

For more detailed instructions, see the full [README.md](README.md) and [scripts.md](scripts.md).
