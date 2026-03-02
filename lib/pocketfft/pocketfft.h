/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see the LICENSE below.
 */

/*! \file pocketfft.h
 *  Public interface of the pocketfft library
 *
 *  Copyright (C) 2008-2018 Max-Planck-Society
 *  \author Martin Reinecke
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. Neither the name of the Max-Planck-Society nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
To make it as explicit as possible that this code is copied from:
https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/raw/master/pocketfft.h?ref_type=heads
No attempt to make it consistent with TerseTS has been made.
*/


#ifndef POCKETFFT_H
#define POCKETFFT_H

#include <stdlib.h>

struct cfft_plan_i;
typedef struct cfft_plan_i * cfft_plan;
cfft_plan make_cfft_plan (size_t length);
void destroy_cfft_plan (cfft_plan plan);
int cfft_backward(cfft_plan plan, double c[], double fct);
int cfft_forward(cfft_plan plan, double c[], double fct);
size_t cfft_length(cfft_plan plan);

struct rfft_plan_i;
typedef struct rfft_plan_i * rfft_plan;
rfft_plan make_rfft_plan (size_t length);
void destroy_rfft_plan (rfft_plan plan);
int rfft_backward(rfft_plan plan, double c[], double fct);
int rfft_forward(rfft_plan plan, double c[], double fct);
size_t rfft_length(rfft_plan plan);

#endif
