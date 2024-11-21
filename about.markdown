---
layout: default
title: Blogs
permalink: /about/
---

# Blog
Welcome to my blog! Here are some of my recent posts:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>


