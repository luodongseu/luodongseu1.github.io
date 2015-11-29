---
layout: post
title: chromium os概述(一)
category: android
keywords: android,浏览器,chromium,chrome
---



## 0x01. chromium概述


### 0x011 chromium历史
WebKit是一个开源的项目,其前身是来源于KDE的KHTML和KJS.该项目核心为支持w3c标准的浏览器渲染,具有跨平台,接口完整等特点.外部扩展package功能完整.使得开发者开发一款支持完整w3c标准的浏览器变得十分容易.
本文重点不是讲述chromium渲染核心,而是讲述chrome扩展层,即**具体android平台的java层移植代码**.

chromium出现在android的版本是4.4KitKat,api19,webview将其驱动从webkit全部更换成了chromium.在此之前chromium一直是独立于aosp,之后chromium将部分java插口代码迁入aosp.

但迁入aosp的chromium在java层始终是部分开源的.chrome在那时也同样是闭源的,也就意味着开发者看不到chrome java层的实现,只有渲染层&驱动层的代码.

之后chrome开源了,chromium java层也逐渐开源了.值得一提的是,chrome和chromium在4.4乃至5.0还是有区别的.也就是说chrome不是完全移植于chromium.直到后来的版本合并chrome才成了chromium的移植产物.

### 0x012 资源下载


chromium git : https://chromium.googlesource.com/chromium/src.git/

由于国内网你懂得,源码8个多g ,git clone 显然不现实,很容易break.
博主通过特殊手段下载了一份(花了很大精力)放在国内云盘,版本是22aa445bafda8fbf2e4436327bc2ecc68902a15b(Wed Nov 25 13:49:22 2015)开发版.

连接:http://pan.baidu.com/s/1bngEzq3


## 0x02. chromium源码概论


## 0x02. 理解chromium&webview&chrome



未完待续