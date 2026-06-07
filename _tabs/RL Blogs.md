---
title: RL Blogs
icon: fas fa-robot
order: 6
math: true
---

Reinforcement learning is one of the most beautiful meeting points of probability, optimization, dynamic programming, stochastic approximation, and control. At its core, RL asks a deceptively simple question: how should an agent learn to make good decisions from interaction?

This page collects my notes on reinforcement learning and related areas. Some posts are meant to build foundations from the ground up: value functions, Bellman equations, temporal-difference learning, Q-learning, function approximation, and policy-gradient ideas. Others are more research-oriented, focusing on robustness, adversarial corruption, Markovian data, decentralized learning, and the technical ideas that shape modern RL theory. I also use this space to write about papers that influenced how I think about learning algorithms. The goal is not only to summarize results, but to explain the mathematical intuition behind them: what problem the paper solves, why the proof technique is interesting, and how the idea connects to broader questions in reliable and robust learning.

---

{% assign rl_posts = site.categories["rl-blogs"] | sort: "date" | reverse %}

## RL Fundamentals Blogs

<ul>
{% for post in rl_posts %}
  {% if post.rl_section == "rl-fundamentals" %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%B %d, %Y" }}</small>
  </li>
  {% endif %}
{% endfor %}
</ul>

---

## Adversarially-Robust RL Blogs

<ul>
{% for post in rl_posts %}
  {% if post.rl_section == "robust-rl" %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%B %d, %Y" }}</small>
  </li>
  {% endif %}
{% endfor %}
</ul>

---

## Papers That Shaped My Research

<ul>
{% for post in rl_posts %}
  {% if post.rl_section == "research-papers" %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%B %d, %Y" }}</small>
  </li>
  {% endif %}
{% endfor %}
</ul>

---

## Adversarial Robustness in Machine Learning

<ul>
{% for post in rl_posts %}
  {% if post.rl_section == "adversarial-ml" %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%B %d, %Y" }}</small>
  </li>
  {% endif %}
{% endfor %}
</ul>
