// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_NOTIFICATIONS_NOTIFICATION_COMMON_H_
#define CHROME_BROWSER_NOTIFICATIONS_NOTIFICATION_COMMON_H_

#include "base/feature_list.h"
#include "chrome/browser/notifications/notification_handler.h"
#include "url/gurl.h"

class GURL;
class Profile;

// Shared functionality for both in page and persistent notification
class NotificationCommon {
 public:
  // Things as user can do to a notification.
  // TODO(peter): Prefix these options with OPERATION_.
  enum Operation {
    CLICK = 0,
    CLOSE = 1,
    DISABLE_PERMISSION = 2,
    SETTINGS = 3,
    OPERATION_MAX = SETTINGS
  };

  // A struct that contains extra data about a notification specific to one of
  // the above types.
  struct Metadata {
    virtual ~Metadata();

    NotificationHandler::Type type;
  };

  // Open the Notification settings screen when clicking the right button.
  static void OpenNotificationSettings(Profile* profile, const GURL& origin);
};

// Metadata for PERSISTENT notifications.
struct PersistentNotificationMetadata : public NotificationCommon::Metadata {
  PersistentNotificationMetadata();
  ~PersistentNotificationMetadata() override;

  static const PersistentNotificationMetadata* From(const Metadata* metadata);

  GURL service_worker_scope;
};

#endif  // CHROME_BROWSER_NOTIFICATIONS_NOTIFICATION_COMMON_H_
