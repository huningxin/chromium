// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/views/payments/view_stack.h"

#include <memory>
#include <utility>

#include "base/memory/ptr_util.h"
#include "ui/views/layout/fill_layout.h"

ViewStack::ViewStack()
    : slide_in_animator_(base::MakeUnique<views::BoundsAnimator>(this)),
      slide_out_animator_(base::MakeUnique<views::BoundsAnimator>(this)) {
  SetLayoutManager(std::make_unique<views::FillLayout>());

  slide_out_animator_->AddObserver(this);
  slide_in_animator_->AddObserver(this);

  // Paint to a layer and Mask to Bounds, otherwise descendant views that paint
  // to a layer themselves will still paint while they're being animated out and
  // are out of bounds of their parent.
  SetPaintToLayer();
  layer()->SetMasksToBounds(true);

  slide_in_animator_->set_tween_type(gfx::Tween::FAST_OUT_SLOW_IN);
  slide_out_animator_->set_tween_type(gfx::Tween::FAST_OUT_SLOW_IN);
}

ViewStack::~ViewStack() {}

void ViewStack::Push(std::unique_ptr<views::View> view, bool animate) {
  gfx::Rect destination = bounds();
  destination.set_origin(gfx::Point(0, 0));

  if (animate) {
    // First add the new view out of bounds since it'll slide in from right to
    // left.
    view->SetBounds(width(), 0, width(), height());
  } else {
    view->SetBoundsRect(destination);
  }
  view->Layout();
  AddChildView(view.get());

  view->set_owned_by_client();

  // Add the new view to the stack so it can be popped later when navigating
  // back to the previous screen.
  stack_.push_back(std::move(view));

  if (animate) {
    // Animate the new view to be right on top of this one.
    slide_in_animator_->AnimateViewTo(stack_.back().get(), destination);
  } else {
    // This is handled by the post-animation callback in the animated case, so
    // trigger it synchronously here.
    HideCoveredViews();
    RequestFocus();
  }
}

void ViewStack::Pop() {
  DCHECK_LT(1u, size());  // There must be at least one view left after popping.

  gfx::Rect destination = bounds();
  destination.set_origin(gfx::Point(width(), 0));

  // Set the second-to-last view as visible, since it is about to be revealed
  // when the last view animates out.
  stack_[size() - 2]->SetVisible(true);

  slide_out_animator_->AnimateViewTo(
      stack_.back().get(), destination);
}

void ViewStack::PopMany(int n) {
  DCHECK_LT(static_cast<size_t>(n), size());  // The stack can never be empty.

  size_t pre_size = stack_.size();
  // Erase N - 1 elements now, the last one will be erased when its animation
  // completes
  stack_.erase(stack_.end() - n, stack_.end() - 1);
  DCHECK_EQ(pre_size - n + 1, stack_.size());

  Pop();
}

size_t ViewStack::size() const {
  return stack_.size();
}

bool ViewStack::CanProcessEventsWithinSubtree() const {
  return !slide_in_animator_->IsAnimating() &&
         !slide_out_animator_->IsAnimating();
}

void ViewStack::Layout() {
  views::View::Layout();

  // If this view's bounds changed since the beginning of an animation, the
  // animator's targets have to be changed as well.
  gfx::Rect in_new_destination = bounds();
  in_new_destination.set_origin(gfx::Point(0, 0));
  UpdateAnimatorBounds(slide_in_animator_.get(), in_new_destination);


  gfx::Rect out_new_destination = bounds();
  out_new_destination.set_origin(gfx::Point(width(), 0));
  UpdateAnimatorBounds(slide_out_animator_.get(), out_new_destination);
}

void ViewStack::RequestFocus() {
  // The view can only be focused if it has a widget already. It's possible that
  // this isn't the case if some views are pushed before the stack is added to a
  // hierarchy that has a widget.
  if (top()->GetWidget())
    top()->RequestFocus();
}

void ViewStack::HideCoveredViews() {
  // Iterate through all but the last (topmost) view.
  for (size_t i = 0; i + 1 < size(); i++) {
    stack_[i]->SetVisible(false);
  }
}

void ViewStack::UpdateAnimatorBounds(
    views::BoundsAnimator* animator, const gfx::Rect& target) {
  // If an animator is currently animating, figure out which views and update
  // their target bounds.
  if (animator->IsAnimating()) {
    for (auto& view : stack_) {
      if (animator->IsAnimating(view.get())) {
        animator->SetTargetBounds(view.get(), target);
      }
    }
  }
}

void ViewStack::OnBoundsAnimatorDone(views::BoundsAnimator* animator) {
  if (animator == slide_out_animator_.get()) {
    stack_.pop_back();
    DCHECK(!stack_.empty()) << "State stack should never be empty";
  } else if (animator == slide_in_animator_.get()) {
    HideCoveredViews();
  } else {
    NOTREACHED();
  }
  RequestFocus();
}
