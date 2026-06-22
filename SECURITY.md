# Security Policy

## Supported Versions

iw-rag is under active development. Security fixes are applied to the latest
released version on the default branch. Older snapshots are not maintained.

| Version        | Supported          |
| -------------- | ------------------ |
| latest (`main`)| :white_check_mark: |
| older          | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them privately through one of these channels:

- **Preferred:** [GitHub private vulnerability reporting](https://github.com/innovation-ways/iw-rag/security/advisories/new)
  (Security → Report a vulnerability)
- **Email:** info@innovation-ways.com

Please include as much of the following as you can:

- The type of issue (e.g. injection, auth bypass, ACL leak, secret exposure).
- The component affected (e.g. retrieval spine, agent graph, API routes,
  Operator Console, Benchmark Studio, ingestion pipeline).
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept or exploit code, if available.
- The impact, including how an attacker might exploit it.

This information helps us triage your report more quickly.

## Disclosure Process

- We will acknowledge receipt of your report within **3 business days**.
- We aim to provide an initial assessment within **10 business days**.
- We will keep you informed of progress toward a fix and a full announcement,
  and may ask for additional information or guidance.
- We follow a **coordinated disclosure** model: we ask that you give us a
  reasonable opportunity to release a fix before any public disclosure.

## Scope

iw-rag is designed to run **fully locally on a single machine** — the
generator, embedder, reranker, guardrails, and judges are all open-weight
models served on-box, and no data is sent to any hosted API. Reports that are
especially relevant include:

- **Access control (ACL) bypass** — the document-level ACL is a *pre-filter*
  driven by authenticated `principal_groups`; any path that lets a caller read
  chunks outside their groups is in scope.
- **Authentication / bearer-token** weaknesses on the API or Operator Console.
- **Prompt injection** that defeats the fail-closed verification layer or the
  input/output guardrails.
- **Secret or credential exposure** in code, config, or logs.
- **Local privilege or container escape** in the Docker Compose stack.

Because the system is local-only by design, classic remote/network attack
surface is limited — but misconfigurations that expose the API or console
beyond `localhost` are in scope.

Thank you for helping keep iw-rag and its users safe.
