---
layout: default
title: Tech Notes
description: 
---

<section class="home-hero">
  <p class="eyebrow">Tech Notes Archive</p>
  <h1></h1>
  <p></p>
</section>

<section class="post-list" aria-labelledby="latest-posts">
  <h2 class="section-title" id="latest-posts">Latest Posts</h2>
  <div class="post-grid">
    {% for post in site.posts %}
      <a class="post-card" href="{{ post.url | relative_url }}">
        {% if post.image %}
          <img src="{{ post.image | relative_url }}" alt="{{ post.title }} 대표 이미지">
        {% endif %}
        <div class="post-card-body">
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
          <h2>{{ post.title }}</h2>
          <p>{{ post.description }}</p>
        </div>
      </a>
    {% endfor %}
  </div>
</section>
