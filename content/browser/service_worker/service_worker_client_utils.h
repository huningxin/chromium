// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_SERVICE_WORKER_SERVICE_WORKER_CLIENT_UTILS_H_
#define CONTENT_BROWSER_SERVICE_WORKER_SERVICE_WORKER_CLIENT_UTILS_H_

#include <string>
#include <vector>

#include "base/callback.h"
#include "base/memory/weak_ptr.h"
#include "content/common/service_worker/service_worker_status_code.h"
#include "third_party/WebKit/common/service_worker/service_worker_client.mojom.h"
#include "ui/base/mojo/window_open_disposition.mojom.h"

class GURL;

namespace content {

class ServiceWorkerContextCore;
class ServiceWorkerProviderHost;
class ServiceWorkerVersion;

namespace service_worker_client_utils {

using NavigationCallback = base::Callback<void(
    ServiceWorkerStatusCode status,
    const blink::mojom::ServiceWorkerClientInfo& client_info)>;
using ClientCallback = base::Callback<void(
    const blink::mojom::ServiceWorkerClientInfo& client_info)>;
using GetClientCallback = base::OnceCallback<void(
    blink::mojom::ServiceWorkerClientInfoPtr client_info)>;
using ServiceWorkerClientPtrs =
    std::vector<blink::mojom::ServiceWorkerClientInfoPtr>;
using ClientsCallback =
    base::Callback<void(std::unique_ptr<ServiceWorkerClientPtrs> clients)>;

// Focuses the window client associated with |provider_host|. |callback| is
// called with the client information on completion.
void FocusWindowClient(ServiceWorkerProviderHost* provider_host,
                       const ClientCallback& callback);

// Opens a new window and navigates it to |url|. |callback| is called with the
// window's client information on completion.
void OpenWindow(const GURL& url,
                const GURL& script_url,
                int worker_process_id,
                const base::WeakPtr<ServiceWorkerContextCore>& context,
                WindowOpenDisposition disposition,
                const NavigationCallback& callback);

// Navigates the client specified by |process_id| and |frame_id| to |url|.
// |callback| is called with the client information on completion.
void NavigateClient(const GURL& url,
                    const GURL& script_url,
                    int process_id,
                    int frame_id,
                    const base::WeakPtr<ServiceWorkerContextCore>& context,
                    const NavigationCallback& callback);

// Gets the client specified by |provider_host|. |callback| is called with the
// client information on completion.
void GetClient(const ServiceWorkerProviderHost* provider_host,
               GetClientCallback callback);

// Collects clients matched with |options|. |callback| is called with the client
// information sorted in MRU order (most recently focused order) on completion.
void GetClients(const base::WeakPtr<ServiceWorkerVersion>& controller,
                blink::mojom::ServiceWorkerClientQueryOptionsPtr options,
                const ClientsCallback& callback);

}  // namespace service_worker_client_utils

}  // namespace content

#endif  // CONTENT_BROWSER_SERVICE_WORKER_SERVICE_WORKER_CLIENT_UTILS_H_
