# Linear Context

Before planning or editing, fetch the issue's recent comments with
`linear_graphql`, newest last, using the issue id supplied by Symphony.

Use this query shape:

```graphql
query IssueComments($id: String!) {
  issue(id: $id) {
    comments(first: 20) {
      nodes {
        body
        createdAt
        user {
          name
        }
      }
    }
  }
}
```

Treat recent human comments as part of the task context, especially comments
made after the latest completion, blocker, or queue comment.

If a recent human comment asks a clarification question rather than requesting
implementation, answer it in a Linear comment and do not make code changes or
move the issue to `In Review`.

If a recent comment requests rework on an existing PR or branch, inspect the PR
or branch before editing.

Before each progress or completion comment, re-fetch recent comments and
incorporate any new human reply first.
