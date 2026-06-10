---
title: RL Blogs
icon: fas fa-pen-nib 
order: 6
math: true
---

Reinforcement learning is one of the most beautiful meeting points of probability, optimization, dynamic programming, stochastic approximation, and control. At its core, RL asks a deceptively simple question: how should an agent learn to make good decisions from interaction?

This page brings together my notes on reinforcement learning, optimization, and learning under uncertainty. Some posts build the core mathematical foundations of RL; others are closer to my research interests, focusing on robustness, adversarial corruption, Markovian data, decentralized algorithms, and finite-time analysis.

I also use this space to discuss papers that have shaped how I think about learning algorithms. Rather than simply summarizing results, my goal is to unpack the mathematical ideas behind them: what problem is being addressed, why the proof technique matters, where the assumptions enter, and how the result connects to broader questions about reliable and robust decision-making.

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
