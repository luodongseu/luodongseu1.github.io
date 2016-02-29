---
layout: post
title: sougou输入法搜索劫持问题调研
category: android
keywords: android,smali,hijack
---


# 0x01.前言

记于在百度实习的这段时间,这是临时发下来给我的一个case

# 0x02.问题

当使用android搜狗输入法进行搜索的时候,其会有多一层搜索选项.当点击了该层的结果将会被sougou搜索劫持

**现象一**:

![1](http://7xkw0v.com1.z0.glb.clouddn.com/Screenshot_2015-12-22-13-37-00.png)

看一看到搜狗输入法出现了一层劫持搜索

![2](http://7xkw0v.com1.z0.glb.clouddn.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202016-02-28%20%E4%B8%8B%E5%8D%884.25.43.png)

当点击,则被劫持到sougou搜索

![3](http://7xkw0v.com1.z0.glb.clouddn.com/4E74A8C4168B5103D1A32BC1CEB0A300.jpg)


**现象二**:

当在应用宝搜索的时候,搜狗输入法也会出现劫持框

![4](http://7xkw0v.com1.z0.glb.clouddn.com/Screenshot_2015-12-22-14-10-03(1).png)

但是不进行劫持操作而是直接进行搜索

![5](http://7xkw0v.com1.z0.glb.clouddn.com/Screenshot_2015-12-22-14-10-10.png)

# 0x03.分析

## 0x031.劫持的浏览器列表

如果有浏览器劫持发生,一定是用了包名来做判断的

所以可以用包名去作为分析入口.

![6](http://7xkw0v.com1.z0.glb.clouddn.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202016-02-28%20%E4%B8%8B%E5%8D%884.55.30.png)

果然出现了

**com/sogou/upd/webserver/WebServerController**

```

    # 创建一个0xf大小的数组并且放于寄存器v0
    new-array v0, v0, [Ljava/lang/String;

    # 接下来就是初始化数组的过程
    const/4 v1, 0x0

    const-string v2, "com.tencent.mtt"

    aput-object v2, v0, v1

    const/4 v1, 0x1

    const-string v2, "com.android.browser"

    aput-object v2, v0, v1

    const/4 v1, 0x2

    const-string v2, "com.UCMobile"

    aput-object v2, v0, v1

    const/4 v1, 0x3

    const-string v2, "sogou.mobile.explorer"

    aput-object v2, v0, v1

    const/4 v1, 0x4

    const-string v2, "com.baidu.browser.apps"

    aput-object v2, v0, v1

    const/4 v1, 0x5

    const-string v2, "com.android.chrome"

    aput-object v2, v0, v1

    const/4 v1, 0x6

    const-string v2, "org.mozilla.firefox"

    aput-object v2, v0, v1

    const/4 v1, 0x7

    const-string v2, "com.ijinshan.browser_fast"

    aput-object v2, v0, v1

    const/16 v1, 0x8

    const-string v2, "com.ijinshan.browser"

    aput-object v2, v0, v1

    const/16 v1, 0x9

    const-string v2, "com.uc.browser"

    aput-object v2, v0, v1

    const/16 v1, 0xa

    const-string v2, "com.qihoo.browser"

    aput-object v2, v0, v1

    const/16 v1, 0xb

    const-string v2, "com.oupeng.mini.android"

    aput-object v2, v0, v1

    const/16 v1, 0xc

    const-string v2, "com.opera.mini.android"

    aput-object v2, v0, v1

    const/16 v1, 0xd

    const-string v2, "com.mx.browser"

    aput-object v2, v0, v1

    const/16 v1, 0xe

    const-string v2, "com.dolphin.browser.xf"

    aput-object v2, v0, v1

    sput-object v0, Lcom/sogou/upd/webserver/WebServerController;->TARGET_EXPLOER_PACKAGENAME:[Ljava/lang/String;

    # 数组名叫 TARGET_EXPLOER_PACKAGENAME,其初始化完毕

```

分析类WebServerController

1.单例:

```

.method public static getInstance(Landroid/content/Context;)Lcom/sogou/upd/webserver/WebServerController;
    .locals 1

    .prologue
    .line 18
    sget-object v0, Lcom/sogou/upd/webserver/WebServerController;->mController:Lcom/sogou/upd/webserver/WebServerController;

    if-nez v0, :cond_0

    .line 19
    new-instance v0, Lcom/sogou/upd/webserver/WebServerController;

    invoke-direct {v0, p0}, Lcom/sogou/upd/webserver/WebServerController;-><init>(Landroid/content/Context;)V

    sput-object v0, Lcom/sogou/upd/webserver/WebServerController;->mController:Lcom/sogou/upd/webserver/WebServerController;

    .line 21
    :cond_0
    sget-object v0, Lcom/sogou/upd/webserver/WebServerController;->mController:Lcom/sogou/upd/webserver/WebServerController;

    return-object v0
.end method

```

2.