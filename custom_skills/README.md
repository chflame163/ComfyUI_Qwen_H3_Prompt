# Custom H3 Skills

Place one prompt-only Skill in each subdirectory. The directory is discovered when
ComfyUI loads this custom node, so restart ComfyUI after adding or changing a Skill.

Each Skill must contain a `SKILL.md`. The optional YAML front matter may provide
`name`, `description`, `display-name`, and `version`; `meta.yaml` may provide the
same metadata plus tags. The Skill ID must contain only lowercase letters, numbers,
periods, underscores, and hyphens, and must not be `auto`.

Example:

```text
custom_skills/
  cinematic-food/
    SKILL.md
    references/
      shot-guidelines.md
    meta.yaml
```

The node reads Markdown and text references as model guidance. It does not execute
scripts, tools, network calls, Canvas workflows, or approval gates declared by a
third-party Skill. The selected Skill may return an H3 prompt or any other content;
there is no node-level H3 field or formatting requirement.

The included `example_custom_skill/` is a minimal discovery test. Select
`example_custom_skill` in the node to confirm that the custom directory is visible
and its instructions are loaded. Its requested fixed output is
`This is a example of custom_skill.`
