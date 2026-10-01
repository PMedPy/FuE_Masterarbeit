---
title: "{{title}}"
authors: {% for a in creators %}{{a.firstName}} {{a.lastName}}{% if not loop.last %}, {% endif %}{% endfor %}
year: {{date | format("YYYY")}}
citekey: {{citekey}}
tags: [literatur]
---

# {{title}}

**Autor(en):** {% for a in creators %}{{a.firstName}} {{a.lastName}}{% if not loop.last %}, {% endif %}{% endfor %}
**Jahr:** {{date | format("YYYY")}}
**Citekey:** [[@{{citekey}}]]

{% if abstractNote %}
## Abstract
{{abstractNote}}
{% endif %}

## Annotationen

{% persist "annotations" %}
{%- set annotations = annotations | filterby("date", "dateafter", lastImportDate) -%}
{%- if annotations.length > 0 %}
### Importiert am {{importDate | format("YYYY-MM-DD HH:mm")}}

{% for a in annotations -%}
{%- if a.annotatedText %}
> {{a.annotatedText}} [page. {{a.page}}](zotero://open-pdf/library/items/{{a.attachment.itemKey}}?page={{a.page}}&annotation={{a.id}})
{%- endif %}
{%- if a.imageRelativePath %}
![[{{a.imageRelativePath}}]]
{%- endif %}
{%- if a.comment %}

**Notiz:** {{a.comment}}
{%- endif %}

---
{% endfor -%}
{%- endif -%}
{% endpersist %}