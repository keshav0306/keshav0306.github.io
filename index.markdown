---
layout: default
title: Blogssss
---
# Blog

Welcome to my blog! I plan to write blogs on interesting papers that I have read for strengthening my own understanding and for others, but this is some other page:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>
