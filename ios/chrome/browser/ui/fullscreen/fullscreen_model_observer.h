// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IOS_CHROME_BROWSER_UI_FULLSCREEN_FULLSCREEN_MODEL_OBSERVER_H_
#define IOS_CHROME_BROWSER_UI_FULLSCREEN_FULLSCREEN_MODEL_OBSERVER_H_

#include <CoreGraphics/CoreGraphics.h>

#include "base/macros.h"

class FullscreenModel;

// Interface for listening to FullscreenModel changes.
class FullscreenModelObserver {
 public:
  FullscreenModelObserver() = default;
  virtual ~FullscreenModelObserver() = default;

  // Invoked when |model|'s calculated progress() value is updated.
  virtual void FullscreenModelProgressUpdated(FullscreenModel* model) {}

  // Invoked when |model| is enabled or disabled.
  virtual void FullscreenModelEnabledStateChanged(FullscreenModel* model) {}

  // Invoked when a scroll event being tracked by |model| has started.
  virtual void FullscreenModelScrollEventStarted(FullscreenModel* model) {}

  // Invoked when a scroll event being tracked by |model| has ended.
  virtual void FullscreenModelScrollEventEnded(FullscreenModel* model) {}

 private:
  DISALLOW_COPY_AND_ASSIGN(FullscreenModelObserver);
};

#endif  // IOS_CHROME_BROWSER_UI_FULLSCREEN_FULLSCREEN_MODEL_OBSERVER_H_
