---
name: example_custom_skill
description: Return a fixed sentence to verify that a custom Skill was discovered, selected, and loaded.
compatibility: Prompt-only demonstration Skill; no tools, scripts, network access, or external assets are required.
display-name: Example Custom Skill
version: 1.0.0
tags: [example, test, custom-skill]
---

# Example Custom Skill

This is a deliberately minimal custom Skill used to verify the `custom_skills`
directory integration. Its architecture is complete, but it has no production
workflow, asset requirements, tools, or external references.

## Input

Ignore the user request and all connected media for this demonstration.

## Processing

Do not analyze, transform, summarize, or expand the input. Do not ask a question
and do not request confirmation.

## Output

Return exactly this sentence and nothing else:

This is a example of custom_skill.

Do not add Markdown fences, a heading, explanation, H3 fields, punctuation, or a
trailing comment. Preserve the sentence exactly, including its spelling and
underscore.

## Tools and references

No tools, scripts, network calls, Canvas operations, or additional references are
used by this Skill.
