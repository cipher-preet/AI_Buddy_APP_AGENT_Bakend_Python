# 🚀 Enterprise AI Platform

### FastAPI + LangGraph + MCP + Multi-Agent AI Architecture

Enterprise-grade scalable AI agent platform built using:

* FastAPI
* LangGraph
* LangChain
* MCP (Model Context Protocol)
* RAG Architecture
* Multi-Agent Systems
* Redis + PostgreSQL + Qdrant
* Docker + Kubernetes

Designed for:

* ERP AI Systems
* Autonomous AI Agents
* AI Workflow Automation
* Tool Calling Systems
* Enterprise AI Infrastructure

---

## ✨ Features

* ⚡ FastAPI scalable backend
* 🧠 LangGraph workflow orchestration
* 🤖 Multi-agent architecture
* 🔌 MCP tool integration
* 📚 RAG pipelines
* 🧠 Long-term + short-term memory
* 🐳 Docker support
* ☸ Kubernetes-ready infrastructure
* 📈 Observability & monitoring
* 🔄 Async workers & queues
* 🛡 Enterprise-grade architecture

---

## 🏗 High-Level Architecture

```text
Frontend
   ↓
API Gateway
   ↓
Agent Runtime
   ↓
LangGraph Workflow
   ↓
Planner Agent
   ↓
Retriever Agent
   ↓
Memory Layer
   ↓
Execution Agent
   ↓
MCP Client
   ↓
MCP Servers
   ↓
External APIs / ERP / Slack / GitHub
```

---

## 🛠 Tech Stack

| Layer           | Technology     |
| --------------- | -------------- |
| API Framework   | FastAPI        |
| Workflow Engine | LangGraph      |
| AI Framework    | LangChain      |
| Tool Protocol   | MCP            |
| Database        | PostgreSQL     |
| Cache           | Redis          |
| Vector DB       | Qdrant         |
| Queue System    | Kafka / Celery |
| Containers      | Docker         |
| Orchestration   | Kubernetes     |
| Monitoring      | LangSmith      |

---

## 📂 Project Structure

## FastAPI + LangGraph + MCP + Multi-Agent Architecture

---

# Overview

This project is a scalable enterprise AI platform architecture designed for:

* AI Agents
* ERP Automation
* Multi-Agent Workflows
* MCP Integration
* LangGraph Orchestration
* RAG Systems
* Tool Calling
* Workflow Automation
* Human-in-the-loop Systems
* Real-time AI Execution

---

# Tech Stack

| Layer               | Technology     |
| ------------------- | -------------- |
| API Framework       | FastAPI        |
| Agent Orchestration | LangGraph      |
| LLM Framework       | LangChain      |
| Tool Protocol       | MCP            |
| Database            | PostgreSQL     |
| Cache               | Redis          |
| Vector DB           | Qdrant         |
| Queue               | Kafka / Celery |
| Containerization    | Docker         |
| Orchestration       | Kubernetes     |
| Monitoring          | LangSmith      |

---

# Recommended Python Version

Use:

```bash
Python 3.11
```

Avoid:

```bash
Python 3.12+
```

because some AI libraries may still have compatibility issues.

---

# Create Project

```bash
mkdir enterprise-ai-platform
cd enterprise-ai-platform
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

# Upgrade Pip

```bash
python -m pip install --upgrade pip
```

---

# Install Core Dependencies

```bash
pip install fastapi uvicorn
pip install langgraph
pip install langchain
pip install openai
pip install anthropic
pip install google-generativeai
pip install python-dotenv
pip install sqlalchemy
pip install asyncpg
pip install redis
pip install qdrant-client
pip install celery
pip install aiokafka
pip install loguru
pip install prometheus-client
pip install opentelemetry-api
pip install black
pip install isort
pip install pytest
pip install mypy
```

---

# Full Folder Structure Creation Commands (Windows)

```bash
:: =========================================
:: ENTER PROJECT
:: =========================================

mkdir enterprise-ai-platform
cd enterprise-ai-platform


:: =========================================
:: ROOT FOLDERS
:: =========================================

mkdir apps
mkdir services
mkdir packages
mkdir infrastructure
mkdir tests
mkdir docs
mkdir scripts


:: =========================================
:: APPS
:: =========================================

mkdir apps\api-gateway
mkdir apps\agent-runtime
mkdir apps\worker
mkdir apps\websocket-server
mkdir apps\admin-dashboard


:: =========================================
:: API GATEWAY
:: =========================================

mkdir apps\api-gateway\src
mkdir apps\api-gateway\src\routes
mkdir apps\api-gateway\src\controllers
mkdir apps\api-gateway\src\middleware
mkdir apps\api-gateway\src\websocket
mkdir apps\api-gateway\src\dependencies
mkdir apps\api-gateway\src\config
mkdir apps\api-gateway\src\utils
mkdir apps\api-gateway\tests


:: =========================================
:: AGENT RUNTIME
:: =========================================

mkdir apps\agent-runtime\src

mkdir apps\agent-runtime\src\agents
mkdir apps\agent-runtime\src\agents\planner
mkdir apps\agent-runtime\src\agents\executor
mkdir apps\agent-runtime\src\agents\validator
mkdir apps\agent-runtime\src\agents\retrieval
mkdir apps\agent-runtime\src\agents\supervisor
mkdir apps\agent-runtime\src\agents\memory

mkdir apps\agent-runtime\src\graphs
mkdir apps\agent-runtime\src\graphs\erp_graph
mkdir apps\agent-runtime\src\graphs\support_graph
mkdir apps\agent-runtime\src\graphs\automation_graph
mkdir apps\agent-runtime\src\graphs\common

mkdir apps\agent-runtime\src\nodes
mkdir apps\agent-runtime\src\nodes\planning
mkdir apps\agent-runtime\src\nodes\reasoning
mkdir apps\agent-runtime\src\nodes\execution
mkdir apps\agent-runtime\src\nodes\retrieval
mkdir apps\agent-runtime\src\nodes\approval
mkdir apps\agent-runtime\src\nodes\memory

mkdir apps\agent-runtime\src\state

mkdir apps\agent-runtime\src\tools
mkdir apps\agent-runtime\src\tools\adapters
mkdir apps\agent-runtime\src\tools\wrappers
mkdir apps\agent-runtime\src\tools\registry

mkdir apps\agent-runtime\src\mcp
mkdir apps\agent-runtime\src\mcp\client
mkdir apps\agent-runtime\src\mcp\transport
mkdir apps\agent-runtime\src\mcp\discovery
mkdir apps\agent-runtime\src\mcp\schemas

mkdir apps\agent-runtime\src\memory
mkdir apps\agent-runtime\src\memory\short_term
mkdir apps\agent-runtime\src\memory\long_term
mkdir apps\agent-runtime\src\memory\semantic
mkdir apps\agent-runtime\src\memory\vector
mkdir apps\agent-runtime\src\memory\session

mkdir apps\agent-runtime\src\rag
mkdir apps\agent-runtime\src\rag\ingestion
mkdir apps\agent-runtime\src\rag\chunking
mkdir apps\agent-runtime\src\rag\embeddings
mkdir apps\agent-runtime\src\rag\retrievers
mkdir apps\agent-runtime\src\rag\rerankers
mkdir apps\agent-runtime\src\rag\vectorstores
mkdir apps\agent-runtime\src\rag\pipelines

mkdir apps\agent-runtime\src\llms
mkdir apps\agent-runtime\src\llms\openai
mkdir apps\agent-runtime\src\llms\anthropic
mkdir apps\agent-runtime\src\llms\gemini
mkdir apps\agent-runtime\src\llms\router

mkdir apps\agent-runtime\src\prompts

mkdir apps\agent-runtime\src\observability
mkdir apps\agent-runtime\src\observability\logging
mkdir apps\agent-runtime\src\observability\tracing
mkdir apps\agent-runtime\src\observability\metrics
mkdir apps\agent-runtime\src\observability\langsmith

mkdir apps\agent-runtime\src\security
mkdir apps\agent-runtime\src\config
mkdir apps\agent-runtime\src\constants
mkdir apps\agent-runtime\src\utils

mkdir apps\agent-runtime\tests
```

---

# High-Level Architecture

```text
Frontend
   ↓
API Gateway
   ↓
Agent Runtime
   ↓
LangGraph Workflow
   ↓
Planner Agent
   ↓
Retriever Agent
   ↓
Memory Layer
   ↓
Execution Agent
   ↓
MCP Client
   ↓
MCP Servers
   ↓
External Systems
```

---

# Folder Structure

```text
project-root/
│
├── apps/
│   ├── api-gateway/
│   ├── agent-runtime/
│   ├── worker/
│   ├── websocket-server/
│   └── admin-dashboard/
│
├── services/
│   ├── mcp-servers/
│   ├── rag-service/
│   ├── memory-service/
│   ├── auth-service/
│   ├── notification-service/
│   ├── analytics-service/
│   ├── embedding-service/
│   └── document-service/
│
├── packages/
│   ├── shared/
│   ├── prompts/
│   ├── schemas/
│   ├── sdk/
│   ├── types/
│   └── ui-components/
│
├── infrastructure/
│   ├── docker/
│   ├── kubernetes/
│   ├── terraform/
│   ├── nginx/
│   ├── helm/
│   ├── monitoring/
│   └── github-actions/
│
├── tests/
├── scripts/
├── docs/
│
├── .env
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── Makefile
```

---

# Folder Structure Explained

```text
project-root/
│
├── apps/                                 # Main runnable applications
│
│   ├── api-gateway/                      # Entry point for frontend/mobile requests
│   │   ├── routes/                       # API route definitions
│   │   ├── controllers/                  # Request handlers/business logic
│   │   ├── middleware/                   # Auth, logging, rate limit middleware
│   │   ├── websocket/                    # Real-time socket communication
│   │   ├── dependencies/                 # Dependency injection/services
│   │   ├── config/                       # App configuration/env loading
│   │   ├── utils/                        # Helper functions/utilities
│   │   └── main.py                       # FastAPI application entry point
│   │
│   ├── agent-runtime/                    # Core AI orchestration engine
│   │
│   │   ├── agents/                       # Independent AI agents
│   │   │   ├── planner/                  # Breaks user goals into tasks
│   │   │   ├── executor/                 # Executes tools/actions
│   │   │   ├── validator/                # Validates AI outputs
│   │   │   ├── retrieval/                # Knowledge retrieval agent
│   │   │   ├── supervisor/               # Controls multi-agent workflows
│   │   │   └── memory/                   # Context and memory handling
│   │   │
│   │   ├── graphs/                       # LangGraph workflows
│   │   │   ├── erp_graph/                # ERP-specific workflow graph
│   │   │   ├── support_graph/            # Customer support workflows
│   │   │   ├── automation_graph/         # Automation workflows
│   │   │   └── common/                   # Shared reusable graph logic
│   │   │
│   │   ├── nodes/                        # Reusable graph execution nodes
│   │   │   ├── planning/                 # Planning nodes
│   │   │   ├── reasoning/                # Reasoning/thinking nodes
│   │   │   ├── execution/                # Tool execution nodes
│   │   │   ├── retrieval/                # RAG retrieval nodes
│   │   │   ├── approval/                 # Human approval nodes
│   │   │   └── memory/                   # Memory update nodes
│   │   │
│   │   ├── state/                        # Shared graph state models
│   │   │   ├── graph_state.py            # Main workflow state object
│   │   │   ├── workflow_state.py         # Workflow lifecycle state
│   │   │   └── memory_state.py           # Agent memory state
│   │   │
│   │   ├── tools/                        # Internal tool abstraction layer
│   │   │   ├── adapters/                 # External system adapters
│   │   │   ├── wrappers/                 # Tool wrappers/normalizers
│   │   │   └── registry/                 # Tool registration system
│   │   │
│   │   ├── mcp/                          # MCP communication layer
│   │   │   ├── client/                   # MCP client implementation
│   │   │   ├── transport/                # HTTP/WebSocket transport
│   │   │   ├── discovery/                # MCP tool discovery system
│   │   │   └── schemas/                  # MCP request/response schemas
│   │   │
│   │   ├── memory/                       # AI memory architecture
│   │   │   ├── short_term/               # Current session memory
│   │   │   ├── long_term/                # Persistent memory
│   │   │   ├── semantic/                 # Semantic memory storage
│   │   │   ├── vector/                   # Vector memory storage
│   │   │   └── session/                  # User conversation sessions
│   │   │
│   │   ├── rag/                          # Retrieval-Augmented Generation
│   │   │   ├── ingestion/                # Document ingestion pipeline
│   │   │   ├── chunking/                 # Text chunking logic
│   │   │   ├── embeddings/               # Embedding generation
│   │   │   ├── retrievers/               # Vector retrieval logic
│   │   │   ├── rerankers/                # Result reranking system
│   │   │   ├── vectorstores/             # Vector DB connectors
│   │   │   └── pipelines/                # Complete RAG pipelines
│   │   │
│   │   ├── llms/                         # LLM provider abstraction
│   │   │   ├── openai/                   # OpenAI integration
│   │   │   ├── anthropic/                # Claude integration
│   │   │   ├── gemini/                   # Gemini integration
│   │   │   └── router/                   # Multi-model routing logic
│   │   │
│   │   ├── prompts/                      # AI prompts/templates
│   │   │
│   │   ├── observability/                # Monitoring and tracing
│   │   │   ├── logging/                  # Structured logging
│   │   │   ├── tracing/                  # Distributed tracing
│   │   │   ├── metrics/                  # Metrics collection
│   │   │   └── langsmith/                # LangSmith integration
│   │   │
│   │   ├── security/                     # Security/auth logic
│   │   ├── config/                       # Runtime configurations
│   │   ├── constants/                    # Global constants
│   │   ├── utils/                        # Shared helper utilities
│   │   └── main.py                       # Agent runtime entry point
│   │
│   ├── worker/                           # Background async processing
│   │   ├── queues/                       # Queue definitions
│   │   ├── consumers/                    # Queue consumers/workers
│   │   ├── schedulers/                   # Cron/scheduled jobs
│   │   ├── retry/                        # Retry/failure handling
│   │   ├── jobs/                         # Background job logic
│   │   └── main.py                       # Worker entry point
│   │
│   ├── websocket-server/                 # Dedicated real-time service
│   └── admin-dashboard/                  # Admin frontend/backend
│
├── services/                             # Independent microservices
│
│   ├── mcp-servers/                      # All MCP tool servers
│   │   ├── jira-mcp/                     # Jira integration server
│   │   ├── slack-mcp/                    # Slack integration server
│   │   ├── github-mcp/                   # GitHub integration server
│   │   ├── gmail-mcp/                    # Gmail integration server
│   │   ├── database-mcp/                 # Database query server
│   │   ├── erp-mcp/                      # ERP integration server
│   │   └── filesystem-mcp/               # File operation server
│   │
│   ├── rag-service/                      # Standalone RAG service
│   ├── memory-service/                   # Centralized memory service
│   ├── auth-service/                     # Authentication/authorization
│   ├── notification-service/             # Notifications/emails/webhooks
│   ├── analytics-service/                # Analytics and reporting
│   ├── embedding-service/                # Embedding generation service
│   └── document-service/                 # OCR/document parsing
│
├── packages/                             # Shared reusable packages
│
│   ├── shared/                           # Common utilities/helpers
│   │   ├── logging/                      # Shared logging utilities
│   │   ├── exceptions/                   # Custom exception handling
│   │   ├── constants/                    # Shared constants
│   │   ├── validators/                   # Shared validation logic
│   │   ├── helpers/                      # Common helper functions
│   │   └── enums/                        # Global enums
│   │
│   ├── prompts/                          # Versioned centralized prompts
│   │   ├── system/                       # System prompts
│   │   ├── planner/                      # Planner prompts
│   │   ├── executor/                     # Execution prompts
│   │   ├── retrieval/                    # Retrieval prompts
│   │   ├── validation/                   # Validation prompts
│   │   └── templates/                    # Prompt templates
│   │
│   ├── schemas/                          # Shared schemas/types
│   │   ├── api/                          # API request schemas
│   │   ├── agent/                        # Agent schemas
│   │   ├── workflow/                     # Workflow schemas
│   │   ├── tool/                         # Tool schemas
│   │   ├── memory/                       # Memory schemas
│   │   └── mcp/                          # MCP schemas
│   │
│   ├── sdk/                              # Internal SDK/client packages
│   ├── types/                            # Global type definitions
│   └── ui-components/                    # Shared frontend components
│
├── infrastructure/                       # Deployment and infra configs
│
│   ├── docker/                           # Docker configurations
│   ├── kubernetes/                       # Kubernetes manifests
│   ├── terraform/                        # Infrastructure as code
│   ├── nginx/                            # Reverse proxy configs
│   ├── helm/                             # Helm charts
│   ├── monitoring/                       # Grafana/Prometheus configs
│   └── github-actions/                   # CI/CD workflows
│
├── tests/                                # Global testing
│   ├── integration/                      # Integration tests
│   ├── e2e/                              # End-to-end tests
│   ├── load/                             # Load/performance testing
│   └── unit/                             # Unit tests
│
├── scripts/                              # DevOps/helper scripts
│   ├── setup/                            # Environment setup scripts
│   ├── migration/                        # Database migrations
│   ├── deployment/                       # Deployment automation
│   └── seeding/                          # Database seeding scripts
│
├── docs/                                 # Project documentation
│   ├── architecture/                     # Architecture diagrams/docs
│   ├── api/                              # API documentation
│   ├── workflows/                        # Workflow documentation
│   ├── agents/                           # Agent behavior docs
│   └── deployment/                       # Deployment guides
│
├── .env                                  # Environment variables
├── .env.example                          # Example env template
├── docker-compose.yml                    # Local docker orchestration
├── pnpm-workspace.yaml                   # Monorepo workspace config
├── turbo.json                            # Turborepo config
├── Makefile                              # Common automation commands
├── README.md                             # Main project documentation
└── pyproject.toml                        # Python dependency/config


---
# Folder Descriptions

## apps/

Contains all runnable applications.

### api-gateway/

Handles:

* authentication
* REST APIs
* websocket communication
* request routing
* middleware

### agent-runtime/

Core AI orchestration engine.

Contains:

* LangGraph workflows
* agents
* memory
* RAG
* MCP client
* tool execution

### worker/

Handles background jobs.

Examples:

* embeddings
* OCR
* indexing
* scheduled jobs
* retries

---

# Agent Runtime Architecture

## agents/

Contains all AI agents.

### planner/

Breaks user goals into tasks.

### executor/

Executes tools and workflows.

### validator/

Validates AI responses.

### supervisor/

Controls multi-agent execution.

### retrieval/

Handles knowledge retrieval.

### memory/

Manages AI memory.

---

# Graphs

## graphs/

Contains LangGraph workflows.

Examples:

* ERP workflows
* support workflows
* automation workflows
* approval workflows

---

# MCP Architecture

## mcp/

Handles MCP communication.

### client/

Calls MCP tools.

### discovery/

Discovers available tools.

### transport/

Handles communication transport.

### schemas/

MCP request/response schemas.

---

# RAG Architecture

## rag/

Contains complete RAG pipelines.

### ingestion/

Document ingestion.

### chunking/

Text chunking.

### embeddings/

Embedding generation.

### retrievers/

Vector retrieval.

### rerankers/

Result reranking.

### vectorstores/

Database connectors.

---

# Memory Architecture

## memory/

AI memory system.

### short_term/

Current conversation memory.

### long_term/

Persistent memory.

### semantic/

Knowledge memory.

### vector/

Vector-based memory.

### session/

Session memory.

---

# MCP Servers

## services/mcp-servers/

Each external system should have isolated MCP servers.

Examples:

* Jira MCP
* Slack MCP
* GitHub MCP
* Gmail MCP
* Database MCP

---

# Infrastructure

## infrastructure/

Contains deployment configurations.

### docker/

Docker files.

### kubernetes/

K8s manifests.

### terraform/

Infrastructure as code.

### monitoring/

Prometheus/Grafana configs.

---

# Environment Variables

## .env

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

REDIS_URL=redis://localhost:6379
POSTGRES_URL=
QDRANT_URL=

LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
```

---

# Run FastAPI Server

```bash
uvicorn apps.api-gateway.main:app --reload
```

---

# Open API Docs

```text
http://127.0.0.1:8000/docs
```

---

# Docker Compose Example

```yaml
version: '3.9'

services:

  redis:
    image: redis:latest
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ai_platform
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

---

# Enterprise Architecture Principles

## 1. Separate Orchestration from Tools

LangGraph should not directly contain integrations.

## 2. Isolated MCP Servers

Every tool should be independently scalable.

## 3. Shared Schemas

Avoid duplicate models.

## 4. Centralized Prompts

Prompts should be reusable and versioned.

## 5. Durable Execution

Use Redis and checkpoints.

## 6. Async Processing

Use queues for long-running tasks.

## 7. Observability

Track:

* latency
* token usage
* failures
* hallucinations
* workflow states

---

# Future Expansion

## Recommended Next Features

* Human approval workflows
* Multi-agent supervisor systems
* Autonomous execution
* Memory compression
* Agent evaluation
* Prompt versioning
* Cost tracking
* Fine-grained permissions
* Tool sandboxing
* Real-time collaboration

---

# Final Enterprise Flow

```text
User
 ↓
Frontend
 ↓
API Gateway
 ↓
Agent Runtime
 ↓
LangGraph Workflow
 ↓
Planner Agent
 ↓
Retriever Agent
 ↓
Memory Layer
 ↓
Execution Agent
 ↓
MCP Client
 ↓
MCP Servers
 ↓
External APIs / ERP / Slack / GitHub
 ↓
Validation Agent
 ↓
Final Response
```

---

# Recommended Enterprise Stack

| Layer         | Technology |
| ------------- | ---------- |
| API           | FastAPI    |
| Workflow      | LangGraph  |
| AI Framework  | LangChain  |
| Tool Layer    | MCP        |
| Cache         | Redis      |
| DB            | PostgreSQL |
| Vector DB     | Qdrant     |
| Queue         | Kafka      |
| Containers    | Docker     |
| Orchestration | Kubernetes |
| Monitoring    | LangSmith  |

---

# License

Enterprise AI Platform Architecture
