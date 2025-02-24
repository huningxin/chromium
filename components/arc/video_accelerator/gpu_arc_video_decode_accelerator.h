// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_ARC_VIDEO_ACCELERATOR_GPU_ARC_VIDEO_DECODE_ACCELERATOR_H_
#define COMPONENTS_ARC_VIDEO_ACCELERATOR_GPU_ARC_VIDEO_DECODE_ACCELERATOR_H_

#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "base/callback_forward.h"
#include "base/threading/thread_checker.h"
#include "components/arc/common/video_decode_accelerator.mojom.h"
#include "gpu/command_buffer/service/gpu_preferences.h"
#include "media/video/video_decode_accelerator.h"

namespace arc {

class ProtectedBufferManager;
class ProtectedBufferHandle;

// GpuArcVideoDecodeAccelerator is executed in the GPU process.
// It takes decoding requests from ARC via IPC channels and translates and
// sends those requests to an implementation of media::VideoDecodeAccelerator.
// It also calls ARC client functions in media::VideoDecodeAccelerator
// callbacks, e.g., PictureReady, which returns the decoded frames back to the
// ARC side. This class manages Reset and Flush requests and life-cycle of
// passed callback for them. They would be processed in FIFO order.

// For each creation request from GpuArcVideoDecodeAcceleratorHost,
// GpuArcVideoDecodeAccelerator will create a new IPC channel.
class GpuArcVideoDecodeAccelerator
    : public mojom::VideoDecodeAccelerator,
      public media::VideoDecodeAccelerator::Client {
 public:
  GpuArcVideoDecodeAccelerator(
      const gpu::GpuPreferences& gpu_preferences,
      ProtectedBufferManager* protected_buffer_manager);
  ~GpuArcVideoDecodeAccelerator() override;

  // Implementation of media::VideoDecodeAccelerator::Client interface.
  void ProvidePictureBuffers(uint32_t requested_num_of_buffers,
                             media::VideoPixelFormat format,
                             uint32_t textures_per_buffer,
                             const gfx::Size& dimensions,
                             uint32_t texture_target) override;
  void PictureReady(const media::Picture& picture) override;
  void DismissPictureBuffer(int32_t picture_buffer_id) override;
  void NotifyEndOfBitstreamBuffer(int32_t bitstream_buffer_id) override;
  void NotifyFlushDone() override;
  void NotifyResetDone() override;
  void NotifyError(media::VideoDecodeAccelerator::Error error) override;

  // mojom::VideoDecodeAccelerator implementation.
  void Initialize(mojom::VideoDecodeAcceleratorConfigPtr config,
                  mojom::VideoDecodeClientPtr client,
                  InitializeCallback callback) override;
  void AllocateProtectedBuffer(
      mojo::ScopedHandle handle,
      uint64_t size,
      AllocateProtectedBufferCallback callback) override;
  void Decode(mojom::BitstreamBufferPtr bitstream_buffer) override;
  void AssignPictureBuffers(uint32_t count) override;
  void ImportBufferForPictureDeprecated(
      int32_t picture_buffer_id,
      mojo::ScopedHandle handle,
      std::vector<VideoFramePlane> planes) override;
  void ImportBufferForPicture(int32_t picture_buffer_id,
                              mojom::HalPixelFormat format,
                              mojo::ScopedHandle handle,
                              std::vector<VideoFramePlane> planes) override;
  void ReusePictureBuffer(int32_t picture_buffer_id) override;
  void Flush(FlushCallback callback) override;
  void Reset(ResetCallback callback) override;

 private:
  using PendingCallback =
      base::OnceCallback<void(mojom::VideoDecodeAccelerator::Result)>;
  static_assert(std::is_same<ResetCallback, PendingCallback>::value,
                "The type of PendingCallback must match ResetCallback");
  static_assert(std::is_same<FlushCallback, PendingCallback>::value,
                "The type of PendingCallback must match FlushCallback");
  using PendingRequest =
      base::OnceCallback<void(PendingCallback, media::VideoDecodeAccelerator*)>;

  class ScopedBitstreamBuffer;

  // Return true if |planes| is valid for |dmabuf_fd|.
  bool VerifyDmabuf(int dmabuf_fd,
                    const std::vector<VideoFramePlane>& planes) const;

  // Initialize GpuArcVDA and create VDA. It returns SUCCESS if they are
  // successful. Otherwise, returns an error status.
  mojom::VideoDecodeAccelerator::Result InitializeTask(
      mojom::VideoDecodeAcceleratorConfigPtr config);

  // Execute all pending requests until a VDA::Reset() request is encountered.
  // When that happens, we need to explicitly wait for NotifyResetDone().
  // before we continue executing subsequent requests.
  void RunPendingRequests();

  // When |pending_reset_callback_| isn't null, GAVDA is awaiting a preceding
  // Reset() to be finished, and |request| is pended by queueing
  // in |pending_requests_|. Otherwise, the requested VDA operation is executed.
  // In the case of Flush request, the callback is queued to
  // |pending_flush_callbacks_|. In the case of Reset request,
  // the callback is set |pending_reset_callback_|.
  void ExecuteRequest(std::pair<PendingRequest, PendingCallback> request);

  // Requested VDA methods are executed in these functions.
  void FlushRequest(PendingCallback cb, media::VideoDecodeAccelerator* vda);
  void ResetRequest(PendingCallback cb, media::VideoDecodeAccelerator* vda);
  void DecodeRequest(ScopedBitstreamBuffer bitstream_buffer,
                     PendingCallback cb,
                     media::VideoDecodeAccelerator* vda);

  // Global counter that keeps track of the number of active clients (i.e., how
  // many VDAs in use by this class).
  // Since this class only works on the same thread, it's safe to access
  // |client_count_| without lock.
  static size_t client_count_;

  // |error_state_| is true, if GAVDA gets an error from VDA.
  // All the pending functions are cancelled and the callbacks are
  // executed with an error state.
  bool error_state_ = false;

  // The variables for managing callbacks.
  // VDA::Decode(), VDA::Flush() and VDA::Reset() should not be posted to VDA
  // while the previous Reset() hasn't been finished yet (i.e. before
  // NotifyResetDone() is invoked).
  // Those requests will be queued in |pending_requests_| in a FIFO manner,
  // and will be executed once all the preceding Reset() have been finished.
  // |pending_flush_callbacks_| stores all the callbacks corresponding to
  // currently executing Flush()es in VDA. |pending_reset_callback_| is a
  // callback of the currently executing Reset() in VDA.
  // If |pending_flush_callbacks_| is not empty in NotifyResetDone(),
  // as Flush()es may be cancelled by Reset() in VDA, they are called with
  // CANCELLED.
  // In |pending_requests_|, PendingRequest is Reset/Flush/DecodeRequest().
  // PendingCallback is null in the case of Decode().
  // Otherwise, it isn't nullptr and will have to be called eventually.
  std::queue<std::pair<PendingRequest, PendingCallback>> pending_requests_;
  std::queue<FlushCallback> pending_flush_callbacks_;
  ResetCallback pending_reset_callback_;

  gpu::GpuPreferences gpu_preferences_;
  std::unique_ptr<media::VideoDecodeAccelerator> vda_;
  mojom::VideoDecodeClientPtr client_;

  gfx::Size coded_size_;
  gfx::Size pending_coded_size_;
  media::VideoPixelFormat output_pixel_format_ = media::PIXEL_FORMAT_UNKNOWN;

  // Owned by caller.
  ProtectedBufferManager* const protected_buffer_manager_;

  bool secure_mode_ = false;
  size_t output_buffer_count_ = 0;

  // Although the stored data are not used for anything in GAVDA,
  // storing them in GAVDA is needed to keep the protected buffers and
  // their mappings valid on behalf of the GAVDA client.
  // For both of them, it is prohibited to allocate more than the predetermined
  // number of protected buffers.
  // The predetermined number is a constant value (kMaxProtectedInputBuffers)
  // for protected input buffers, and it is |output_buffer_count_| set in
  // AssignPictureBuffers() for protected output buffers.
  // |protected_output_handles_| is indexed by |picture_buffer_id| in
  // ImportBufferForPicture().
  std::vector<std::unique_ptr<ProtectedBufferHandle>> protected_input_handles_;
  std::vector<std::unique_ptr<ProtectedBufferHandle>> protected_output_handles_;

  THREAD_CHECKER(thread_checker_);
  DISALLOW_COPY_AND_ASSIGN(GpuArcVideoDecodeAccelerator);
};

}  // namespace arc

#endif  // COMPONENTS_ARC_VIDEO_ACCELERATOR_GPU_ARC_VIDEO_DECODE_ACCELERATOR_H_
