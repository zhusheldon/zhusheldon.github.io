---
layout: single
title:  "What are people doing with their time? - Part 1"
date:   2025-12-05 00:00:00 -2300
categories: Productivity
---

If I ask you, "What did you do yesterday?",  what would you say? 

Most people would give a vague overview of 1 or 2 things they did, some would say "Not much.", others would go really in-depth into one specific activity. But almost no one is able to give a time-by-by outline of how they spent their day. 

Which is ironic to think about when there are so many calendar and planning tools for people to plan out their day (and some people will plan their day out viligantly) but during active recall, people can't seem to remember.

Furthermore, even when people do happen to plan their day, it almost never goes to plan. Unexpected events arise, plans get pushed back, traffic makes you late. All of these make your expectation of time management turn out differently in reality, and there needs to be better ways to actively track this.

While there are digital tools that can help remedy this behavior, first I propose a data model for such a tool that can help showcase the relationship between a user, their schedule, and their ideal schedule. 

![TimeTrackingSchema]({{ site.url }}{{ site.baseurl }}/assets/images/Time Tracking.png)

With this, we can perform analysis on how users are spending most of their time on by category. We can also view how users are frequently spending their time by different cuts of day and how their actualized time allocation compares to their idealized time allocation. This view will enable users to start reflecting on how to adjust their behavior to better reflect their intentions.

One key aspect I am looking to explore is capturing psychological aspects of the user. Their emotional state, personality, behavioral dispositions, etc. can all lend to their perspective of time, discipline, and motivation toward time optimization.

In later parts, I will discuss how to implement this in a web app, what other factors can influence suboptimal time allocation, and building out our analytics warehouse.