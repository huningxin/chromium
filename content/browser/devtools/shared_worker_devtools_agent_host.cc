// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/devtools/shared_worker_devtools_agent_host.h"

#include "content/browser/devtools/devtools_session.h"
#include "content/browser/devtools/protocol/inspector_handler.h"
#include "content/browser/devtools/protocol/network_handler.h"
#include "content/browser/devtools/protocol/protocol.h"
#include "content/browser/devtools/protocol/schema_handler.h"
#include "content/browser/devtools/shared_worker_devtools_manager.h"
#include "content/browser/shared_worker/shared_worker_host.h"
#include "content/browser/shared_worker/shared_worker_instance.h"
#include "content/browser/shared_worker/shared_worker_service_impl.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/render_process_host.h"

namespace content {

SharedWorkerDevToolsAgentHost::SharedWorkerDevToolsAgentHost(
    SharedWorkerHost* worker_host,
    const base::UnguessableToken& devtools_worker_token)
    : DevToolsAgentHostImpl(devtools_worker_token.ToString()),
      state_(WORKER_NOT_READY),
      worker_host_(worker_host),
      devtools_worker_token_(devtools_worker_token),
      instance_(new SharedWorkerInstance(*worker_host->instance())) {
  NotifyCreated();
}

SharedWorkerDevToolsAgentHost::~SharedWorkerDevToolsAgentHost() {
  SharedWorkerDevToolsManager::GetInstance()->AgentHostDestroyed(this);
}

BrowserContext* SharedWorkerDevToolsAgentHost::GetBrowserContext() {
  RenderProcessHost* rph = GetProcess();
  return rph ? rph->GetBrowserContext() : nullptr;
}

std::string SharedWorkerDevToolsAgentHost::GetType() {
  return kTypeSharedWorker;
}

std::string SharedWorkerDevToolsAgentHost::GetTitle() {
  return instance_->name();
}

GURL SharedWorkerDevToolsAgentHost::GetURL() {
  return instance_->url();
}

bool SharedWorkerDevToolsAgentHost::Activate() {
  return false;
}

void SharedWorkerDevToolsAgentHost::Reload() {
}

bool SharedWorkerDevToolsAgentHost::Close() {
  if (worker_host_)
    worker_host_->TerminateWorker();
  return true;
}

void SharedWorkerDevToolsAgentHost::AttachSession(DevToolsSession* session) {
  session->SetFallThroughForNotFound(true);
  session->AddHandler(std::make_unique<protocol::InspectorHandler>());
  session->AddHandler(std::make_unique<protocol::NetworkHandler>(GetId()));
  session->AddHandler(std::make_unique<protocol::SchemaHandler>());
  session->SetRenderer(GetProcess(), nullptr);
  if (state_ == WORKER_READY)
    session->AttachToAgent(EnsureAgent());
}

void SharedWorkerDevToolsAgentHost::DetachSession(int session_id) {
  // Destroying session automatically detaches in renderer.
}

bool SharedWorkerDevToolsAgentHost::DispatchProtocolMessage(
    DevToolsSession* session,
    const std::string& message) {
  int call_id = 0;
  std::string method;
  if (session->Dispatch(message, &call_id, &method) !=
      protocol::Response::kFallThrough) {
    return true;
  }

  session->DispatchProtocolMessageToAgent(call_id, method, message);
  session->waiting_messages()[call_id] = {method, message};
  return true;
}

bool SharedWorkerDevToolsAgentHost::Matches(SharedWorkerHost* worker_host) {
  return instance_->Matches(*worker_host->instance());
}

void SharedWorkerDevToolsAgentHost::WorkerReadyForInspection() {
  DCHECK_EQ(WORKER_NOT_READY, state_);
  DCHECK(worker_host_);
  state_ = WORKER_READY;
  for (DevToolsSession* session : sessions()) {
    session->ReattachToAgent(EnsureAgent());
    for (const auto& pair : session->waiting_messages()) {
      int call_id = pair.first;
      const DevToolsSession::Message& message = pair.second;
      session->DispatchProtocolMessageToAgent(call_id, message.method,
                                              message.message);
    }
  }
}

void SharedWorkerDevToolsAgentHost::WorkerRestarted(
    SharedWorkerHost* worker_host) {
  DCHECK_EQ(WORKER_TERMINATED, state_);
  DCHECK(!worker_host_);
  state_ = WORKER_NOT_READY;
  worker_host_ = worker_host;
  for (DevToolsSession* session : sessions())
    session->SetRenderer(GetProcess(), nullptr);
}

void SharedWorkerDevToolsAgentHost::WorkerDestroyed() {
  DCHECK_NE(WORKER_TERMINATED, state_);
  DCHECK(worker_host_);
  state_ = WORKER_TERMINATED;
  for (auto* inspector : protocol::InspectorHandler::ForAgentHost(this))
    inspector->TargetCrashed();
  for (DevToolsSession* session : sessions())
    session->SetRenderer(nullptr, nullptr);
  worker_host_ = nullptr;
  agent_ptr_.reset();
}

RenderProcessHost* SharedWorkerDevToolsAgentHost::GetProcess() {
  return worker_host_ ? RenderProcessHost::FromID(worker_host_->process_id())
                      : nullptr;
}

const blink::mojom::DevToolsAgentAssociatedPtr&
SharedWorkerDevToolsAgentHost::EnsureAgent() {
  DCHECK_EQ(WORKER_READY, state_);
  DCHECK(worker_host_);
  if (!agent_ptr_)
    worker_host_->GetDevToolsAgent(mojo::MakeRequest(&agent_ptr_));
  return agent_ptr_;
}

}  // namespace content
