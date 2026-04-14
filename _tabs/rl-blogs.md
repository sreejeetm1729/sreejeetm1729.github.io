---
title: RL Blogs
icon: fas fa-robot
order: 6
---

# RL Blogs

{% assign rl_posts = site.posts | where_exp: "post", "post.categories contains 'rl-blogs'" | sort: "date" | reverse %}

{% if rl_posts.size > 0 %}
<ul>
  {% for post in rl_posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%B %d, %Y" }}</small>
    </li>
  {% endfor %}
</ul>
{% else %}
<p>No RL blog posts yet.</p>
 {% endif %}
