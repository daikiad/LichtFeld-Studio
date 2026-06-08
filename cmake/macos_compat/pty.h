// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// macOS compatibility shim: glibc exposes openpty()/forkpty() via <pty.h>, but
// on macOS (BSD libc) they live in <util.h>. This shim is only on the include
// path for Apple builds so existing `#include <pty.h>` sites keep working.
#pragma once
#include <util.h>
