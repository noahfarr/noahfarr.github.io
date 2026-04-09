---
layout: page
permalink: /repositories/
title: repositories
description: A list of my GitHub repositories
nav: true
nav_order: 4
---

{% if site.data.repositories.github_repos %}
<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; padding: 0.5rem 0;">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}
