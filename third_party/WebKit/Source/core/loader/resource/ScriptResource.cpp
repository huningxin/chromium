/*
    Copyright (C) 1998 Lars Knoll (knoll@mpi-hd.mpg.de)
    Copyright (C) 2001 Dirk Mueller (mueller@kde.org)
    Copyright (C) 2002 Waldo Bastian (bastian@kde.org)
    Copyright (C) 2006 Samuel Weinig (sam.weinig@gmail.com)
    Copyright (C) 2004, 2005, 2006, 2007, 2008 Apple Inc. All rights reserved.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

    This class provides all functionality needed for loading images, style
    sheets and html pages from the web. It has a memory cache for these objects.
*/

#include "core/loader/resource/ScriptResource.h"

#include "core/dom/Document.h"
#include "core/loader/SubresourceIntegrityHelper.h"
#include "platform/SharedBuffer.h"
#include "platform/instrumentation/tracing/web_memory_allocator_dump.h"
#include "platform/instrumentation/tracing/web_process_memory_dump.h"
#include "platform/loader/SubresourceIntegrity.h"
#include "platform/loader/fetch/FetchParameters.h"
#include "platform/loader/fetch/IntegrityMetadata.h"
#include "platform/loader/fetch/ResourceClientWalker.h"
#include "platform/loader/fetch/ResourceFetcher.h"
#include "platform/loader/fetch/TextResourceDecoderOptions.h"
#include "platform/network/mime/MIMETypeRegistry.h"
#include "services/network/public/interfaces/request_context_frame_type.mojom-blink.h"

namespace blink {

ScriptResource* ScriptResource::Fetch(FetchParameters& params,
                                      ResourceFetcher* fetcher,
                                      ResourceClient* client) {
  DCHECK_EQ(params.GetResourceRequest().GetFrameType(),
            network::mojom::RequestContextFrameType::kNone);
  params.SetRequestContext(WebURLRequest::kRequestContextScript);
  return ToScriptResource(
      fetcher->RequestResource(params, ScriptResourceFactory(), client));
}

ScriptResource::ScriptResource(
    const ResourceRequest& resource_request,
    const ResourceLoaderOptions& options,
    const TextResourceDecoderOptions& decoder_options)
    : TextResource(resource_request, kScript, options, decoder_options) {}

ScriptResource::~ScriptResource() = default;

void ScriptResource::OnMemoryDump(WebMemoryDumpLevelOfDetail level_of_detail,
                                  WebProcessMemoryDump* memory_dump) const {
  Resource::OnMemoryDump(level_of_detail, memory_dump);
  const String name = GetMemoryDumpName() + "/decoded_script";
  auto* dump = memory_dump->CreateMemoryAllocatorDump(name);
  dump->AddScalar("size", "bytes", source_text_.CharactersSizeInBytes());
  memory_dump->AddSuballocation(
      dump->Guid(), String(WTF::Partitions::kAllocatedObjectPoolName));
}

const String& ScriptResource::SourceText() {
  DCHECK(IsLoaded());

  if (source_text_.IsNull() && Data()) {
    String source_text = DecodedText();
    ClearData();
    SetDecodedSize(source_text.CharactersSizeInBytes());
    source_text_ = AtomicString(source_text);
  }

  return source_text_;
}

void ScriptResource::DestroyDecodedDataForFailedRevalidation() {
  source_text_ = AtomicString();
  SetDecodedSize(0);
}

AccessControlStatus ScriptResource::CalculateAccessControlStatus() const {
  if (GetCORSStatus() == CORSStatus::kServiceWorkerOpaque)
    return kOpaqueResource;

  if (IsSameOriginOrCORSSuccessful())
    return kSharableCrossOrigin;

  return kNotSharableCrossOrigin;
}

bool ScriptResource::CanUseCacheValidator() const {
  // Do not revalidate until ClassicPendingScript is removed, i.e. the script
  // content is retrieved in ScriptLoader::ExecuteScriptBlock().
  // crbug.com/692856
  if (HasClientsOrObservers())
    return false;

  return Resource::CanUseCacheValidator();
}

}  // namespace blink
