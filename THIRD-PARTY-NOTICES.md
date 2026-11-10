# Third-party notices

Geometrikal.OpenCV.Managed contains C# ports of code from the projects listed below.
The [NOTICE](NOTICE) file lists the individual source files that each part of
the library was ported from.

## OpenCV

- Source: https://github.com/opencv/opencv (version 5.0.0, included as the
  git submodule at `src/OpenCV`)
- License: Apache License, Version 2.0. The full text is in [LICENSE](LICENSE),
  which is a verbatim copy of the OpenCV `LICENSE` file.
- Copyright: see [licenses/OpenCV-COPYRIGHT](licenses/OpenCV-COPYRIGHT), which
  is a verbatim copy of the OpenCV `COPYRIGHT` file.

## opencv_contrib

- Source: https://github.com/opencv/opencv_contrib (`modules/ximgproc`, guided
  filter)
- License: Apache License, Version 2.0.

## Microsoft C++ Standard Library

`Core/Internal/MsvcSort` and `Stitching/Internal/SeamPairSort` reproduce the
element order of `std::sort` from the Microsoft C++ Standard Library.

- Source: https://github.com/microsoft/STL
- License: Apache License, Version 2.0 with LLVM Exception.

## SQPnP

The SQPnP solver (`Calib/Internal/Sqpnp.cs`) is ported from OpenCV's
`modules/geometry/src/sqpnp.cpp`, which is based on code released under the
following license.

```text
BSD 3-Clause License

Copyright (c) 2020, George Terzakis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## P3P

The P3P solver follows Ding et al. (2023) as included in OpenCV from PoseLib
(Viktor Larsson). It is distributed under the OpenCV license terms above.
