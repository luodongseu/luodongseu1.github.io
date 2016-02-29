---
layout: post
title: smali简单插桩
category: android
keywords: android,smali,语法
---

这里简单介绍一下插桩,抛砖引玉

# 0x01.介绍

# 0x02.插桩过程

## 0x021.插桩类封装


为了方便起见,做到参数单一.而减少外部寄存器的依赖,可以封装一个Log类去掉tag参数


```

.class public Lcom/example/asmali_log/L;
.super Ljava/lang/Object;
.source "L.java"


# static fields
.field private static final TAG:Ljava/lang/String; = "smali_in"


# direct methods
.method public constructor <init>()V
    .locals 0

    .prologue
    .line 5
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method

.method public static final d(Ljava/lang/String;)V
    .locals 1
    .param p0, "log"    # Ljava/lang/String;

    .prologue
    .line 22
    const-string v0, "smali_in"

    invoke-static {v0, p0}, Landroid/util/Log;->d(Ljava/lang/String;Ljava/lang/String;)I

    .line 23
    return-void
.end method

.method public static final e(Ljava/lang/String;)V
    .locals 1
    .param p0, "log"    # Ljava/lang/String;

    .prologue
    .line 10
    const-string v0, "smali_in"

    invoke-static {v0, p0}, Landroid/util/Log;->e(Ljava/lang/String;Ljava/lang/String;)I

    .line 11
    return-void
.end method

.method public static final i(Ljava/lang/String;)V
    .locals 1
    .param p0, "log"    # Ljava/lang/String;

    .prologue
    .line 14
    const-string v0, "smali_in"

    invoke-static {v0, p0}, Landroid/util/Log;->i(Ljava/lang/String;Ljava/lang/String;)I

    .line 15
    return-void
.end method

.method public static final w(Ljava/lang/String;)V
    .locals 1
    .param p0, "log"    # Ljava/lang/String;

    .prologue
    .line 18
    const-string v0, "smali_in"

    invoke-static {v0, p0}, Landroid/util/Log;->w(Ljava/lang/String;Ljava/lang/String;)I

    .line 19
    return-void
.end method


```

注意将该类的路径进行更改,并且在调用的过程中路径问题


## 0x022.进行插桩

源程序:

```
.class public Lcom/example/asmali_log/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"


# static fields
.field private static final str:Ljava/lang/String; = "Hello"


# direct methods
.method public constructor <init>()V
    .locals 0

    .prologue
    .line 9
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V

    return-void
.end method


# virtual methods
.method protected onCreate(Landroid/os/Bundle;)V
    .locals 1
    .param p1, "savedInstanceState"    # Landroid/os/Bundle;

    .prologue
    .line 15
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V

    .line 16
    const/high16 v0, 0x7f030000

    invoke-virtual {p0, v0}, Lcom/example/asmali_log/MainActivity;->setContentView(I)V

    .line 17
    return-void
.end method
```

可以看到是一个空的Activity类.

**插桩**:


```
.class public Lcom/example/asmali_log/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"


# static fields
.field private static final str:Ljava/lang/String; = "Hello"


# direct methods
.method public constructor <init>()V
    .locals 0

    .prologue
    .line 9
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V

    return-void
.end method


# virtual methods
.method protected onCreate(Landroid/os/Bundle;)V
    .locals 1
    .param p1, "savedInstanceState"    # Landroid/os/Bundle;

    .prologue
    .line 15
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V

    .line 16
    const/high16 v0, 0x7f030000

    invoke-virtual {p0, v0}, Lcom/example/asmali_log/MainActivity;->setContentView(I)V
    
    const-string v0, "Hello-smali"
        
    invoke-static {v0}, Lcom/example/asmali_log/L;->i(Ljava/lang/String;)V
    invoke-static {v0}, Lcom/example/asmali_log/L;->i(Ljava/lang/String;)V
    invoke-static {v0}, Lcom/example/asmali_log/L;->i(Ljava/lang/String;)V

    .line 17
    return-void
.end method
```
    
可以看到,我在其内插入了三行封装类L的调用

# 0x023.smali重打包&再签名










