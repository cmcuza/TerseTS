// Copyright 2024 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Camel-compression: split integer/decimal parts, delta-encode integers and XOR-compress decimals.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;
const Math = std.math;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const tester = @import("../tester.zig");
const shared_functions = @import("../utilities/shared_functions.zig");


const MAX_DECIMAL_DIGITS = 10;


/// Результат разделения числа на целую и дробную части.
pub const Parts = struct {
    integer: i64,           // целая часть (со знаком)
    decimal: u64,           // дробная часть как целое (0 .. 10^l - 1)
    decimal_digits: u8,     // количество знаков дробной части (l)
    special: bool,          // true если число не может быть представлено (NaN, Inf, слишком большое)
    raw_bits: u64,          // исходные биты для special-случаев
};


// Битовая запись в буфер (ArrayList(u8))
const BitWriter = struct {
    buffer: *ArrayList(u8),
    allocator: Allocator,
    bit_buffer: u8 = 0,
    bit_count: u8 = 0,

    fn init(allocator: Allocator, buffer: *ArrayList(u8)) BitWriter {
        return .{
            .allocator = allocator,
            .buffer = buffer,
        };
    }

    fn writeBits(self: *BitWriter, value: anytype, bits: u8) !void {
        var v = @as(u64, @intCast(value));
        var remaining = bits;
        while (remaining > 0) {
            const take = @min(remaining, 8 - self.bit_count);
            const mask = (v << @as(u3, @intCast(self.bit_count))) & 0xFF;
            self.bit_buffer |= @as(u8, @intCast(mask));
            self.bit_count += take;
            if (self.bit_count == 8) {
                try self.buffer.append(self.allocator, self.bit_buffer);
                self.bit_buffer = 0;
                self.bit_count = 0;
            }
            v >>= @as(u6, @intCast(take));
            remaining -= take;
        }
    }

    fn writeBit(self: *BitWriter, bit: u1) !void {
        try self.writeBits(bit, 1);
    }

    fn finish(self: *BitWriter) !void {
        if (self.bit_count > 0) {
            try self.buffer.append(self.allocator, self.bit_buffer);
        }
    }
};



// ------------------------------------------------------------
// BitReader — чтение битов из буфера байтов
// ------------------------------------------------------------
const BitReader = struct {
    buffer: []const u8,
    byte_index: usize = 0,
    bit_pos: u3 = 0,        // текущий бит внутри байта (0 = младший? но для совместимости с BitWriter)
    // BitWriter записывал биты начиная с младших? В нашей реализации writeBits делала сдвиг влево, накапливая.
    // Для единообразия читаем в том же порядке: сначала биты, записанные в младшие позиции байта.
    // Однако проще реализовать чтение "сначала старшие биты" — но для совместимости с BitWriter выше нужно аккуратно.
    // Перепишем BitReader, предполагая, что байты записаны в порядке, соответствующем writeBits из прошлого ответа.
    // В writeBits: каждый байт заполняется от младших битов к старшим (bit_count увеличивается, сдвиг влево).
    // Читать будем в обратном порядке: сначала извлекаем биты из текущего байта, начиная с младших.
    fn init(buffer: []const u8) BitReader {
        return .{ .buffer = buffer };
    }

    fn readBits(self: *BitReader, bits: u8) !u64 {
        if (bits > 64) return error.InvalidBitsCount;
        var result: u64 = 0;
        var remaining = bits;
        var shift: u7 = 0;
        while (remaining > 0) {
            if (self.byte_index >= self.buffer.len) return error.EndOfStream;
            const take = @min(remaining, 8 - @as(u8, self.bit_pos));
            const mask = (@as(u64, self.buffer[self.byte_index]) >> @as(u3, @intCast(self.bit_pos))) & ((@as(u64, 1) << @as(u6, @intCast(take))) - 1);
            result |= (mask << @as(u6, @intCast(shift)));
            shift += @as(u7, @intCast(take));
            const new_bit_pos = @as(u8, self.bit_pos) + take;
            if (new_bit_pos == 8) {
                self.byte_index += 1;
                self.bit_pos = 0;
            } else {
                self.bit_pos = @as(u3, @intCast(new_bit_pos));
            }
            remaining -= take;
        }
        return result;
    }

    fn readBit(self: *BitReader) !u1 {
        return @as(u1, @intCast(try self.readBits(1)));
    }
};

/// Сжатие целых частей временного ряда согласно Algorithm 1.
/// Параметры:
///   - writer: битовый поток для записи
///   - int_part: целая часть текущего значения (v_t.d_int)
///   - prev_int: целая часть предыдущего значения (v_{t-1}.d_int), игнорируется для t = 1
///   - index: номер элемента (начиная с 1)
pub fn compressInteger(writer: *BitWriter, int_part: i64, prev_int: i64, index: usize) !void {
    if (index == 1) {
        // первое значение: записываем как есть в 64 бита (беззнаковое представление)
        try writer.writeBits(@as(u64, @bitCast(int_part)), 64);
        return;
    }

    const diff = int_part - prev_int;
    const abs_diff = @abs(diff);

    if (abs_diff <= 1) {
        // значения -1, 0, +1 кодируются 2 битами: diff+1 даёт 0,1,2
        const code = @as(u2, @intCast(diff + 1));
        try writer.writeBits(code, 2);
    } else {
        // знак: 1 для положительного, 0 для отрицательного
        const sign_bit: u1 = if (diff >= 0) 1 else 0;
        try writer.writeBit(sign_bit);
        // диапазон: 0 если abs_diff < 8, иначе 1
        const range_bit: u1 = if (abs_diff < 8) 0 else 1;
        try writer.writeBit(range_bit);
        const bits_count: u8 = if (abs_diff < 8) 3 else 16;
        try writer.writeBits(@as(u64, @intCast(abs_diff)), bits_count);
    }
}




fn compressDecimalWithoutL(writer: *BitWriter, v: f64, l: u8) !void { 

    const v_dec = v - @floor(v);  // дробная часть в [0,1)
    const threshold = Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));

    var dxor: f64 = undefined;
    if (v_dec >= threshold) {
        // Флаг = 1
        try writer.writeBit(1);
        dxor = calculateDxor(v_dec, l);
        // XOR двоичных представлений v_dec и dxor
        const v_bits = @as(u64, @bitCast(v_dec));
        const dxor_bits = @as(u64, @bitCast(dxor));
        const xor_result = v_bits ^ dxor_bits;
        // Сдвиг вправо на (52 - l) бит, затем запись младших l бит
        const shifted = xor_result >> @as(u6, @intCast(52 - l));
        const center_bits = shifted & ((@as(u64, 1) << @as(u6, @intCast(l))) - 1);
        try writer.writeBits(center_bits, l);
    } else {
        // Флаг = 0 (в оригинале строка 9 была "out.writeBit(1)", но это, вероятно, опечатка)
        try writer.writeBit(0);
        dxor = v_dec;
    }

    // 2. Преобразуем dxor в целое: dxor' = dxor * 10^l
    const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
    const dxor_prime = @as(u64, @intFromFloat(@round(dxor * scale)));

    // 3. Кодирование dxor' с переменным числом бит в зависимости от l
    if (l <= 1) {
        try writer.writeBits(dxor_prime, l + 1);
    } else if (l == 2) {
        const flag_bit: u1 = if (dxor_prime <= 4) 0 else 1;
        try writer.writeBit(flag_bit);
        const bits_count: u8 = if (dxor_prime <= 4) 2 else 5;
        try writer.writeBits(dxor_prime, bits_count);
    } else {
        // l >= 3
        const max_val = @ceil(Math.log2(Math.pow(f64, 10.0, @as(f64, @floatFromInt(l))) /
                                        Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)))));
        const max = @as(u64, @intFromFloat(max_val));
        // Вычисляем пороги 2^{max/4}, 2^{2*max/4}, 2^{3*max/4}
        const pow2 = Math.pow(f64, 2.0, 1.0);
        _ = pow2;
        const threshold1 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, @as(f64, @floatFromInt(max)) / 4.0)));
        const threshold2 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, 2.0 * @as(f64, @floatFromInt(max)) / 4.0)));
        const threshold3 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, 3.0 * @as(f64, @floatFromInt(max)) / 4.0)));
        const thresholds = [_]u64{ threshold1, threshold2, threshold3 };
        var index: u8 = 3;
        for (thresholds, 0..) |th, i| {
            if (dxor_prime <= th) {
                index = @as(u8, @intCast(i));
                break;
            }
        }
        // Записываем index как 2 бита
        try writer.writeBits(index, 2);
        const bits_to_write = (index + 1) * max / 4;
        try writer.writeBits(dxor_prime, @as(u8, @intCast(bits_to_write)));
    }
}


/// Вычисляет dxor по теореме 3.1: dxor.dec = v_dec - 2^{-l} * floor(v_dec / 2^{-l})
fn calculateDxor(v_dec: f64, l: u8) f64 {
    const step = Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    const quotient = @floor(v_dec / step + 1e-12);
    return quotient * step;
}

/// Вычисляет количество десятичных знаков (аналог calculateDecimalCount, но используется в splitNumber)
fn computeDecimalDigits(value: f64) u8 {
    return calculateDecimalCount(value);
}


fn calculateDecimalCount(value: f64) u8 {
    const abs_val = @abs(value);
    const int_part = @floor(abs_val);
    const frac = abs_val - int_part;
    if (frac < 1e-12) return 0;
    var count: u8 = 0;
    var remaining = frac;
    while (count < 10 and remaining > 1e-12) {
        remaining *= 10.0;
        const digit = @floor(remaining);
        remaining -= digit;
        count += 1;
        if (@abs(remaining) < 1e-12) break;
    }
    return @min(count, 31); // ограничиваем 31, так как l занимает 5 бит
}

pub fn splitNumber(number: f64, fixed_l: ?u8) Parts {
    // Обработка special значений (NaN, Inf)
    if (!Math.isFinite(number)) {
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    const l = fixed_l orelse computeDecimalDigits(number);
    if (l == 0) {
        // Число целое
        const floored = Math.floor(number);
        const max_i64 = @as(f64, @floatFromInt(Math.maxInt(i64)));
        const min_i64 = @as(f64, @floatFromInt(Math.minInt(i64)));
        if (floored > max_i64 or floored < min_i64) {
            // Слишком большое целое – сохраняем как special
            return .{
                .integer = 0,
                .decimal = 0,
                .decimal_digits = 0,
                .special = true,
                .raw_bits = @as(u64, @bitCast(number)),
            };
        }
        const integer = @as(i64, @intFromFloat(floored));
        return .{
            .integer = integer,
            .decimal = 0,
            .decimal_digits = 0,
            .special = false,
            .raw_bits = 0,
        };
    }

    // Масштабируем: умножаем на 10^l, округляем до ближайшего целого
    const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
    const scaled = Math.round(number * scale);
    // Проверка на переполнение i64 (диапазон ~ ±9.22e18)
    const max_i64 = @as(f64, @floatFromInt(Math.maxInt(i64)));
    const min_i64 = @as(f64, @floatFromInt(Math.minInt(i64)));
    if (scaled > max_i64 or scaled < min_i64) {
        // Число слишком велико для представления в i64 -> сохраняем как special
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    const scaled_int = @as(i64, @intFromFloat(scaled));
    // Целая часть: деление на 10^l с округлением к нулю (для отрицательных чисел)
    const integer = @divTrunc(scaled_int, @as(i64, @intFromFloat(scale)));
    // Дробная часть: остаток (всегда неотрицательный)
    const decimal = @abs(scaled_int - integer * @as(i64, @intFromFloat(scale)));

    // Убеждаемся, что дробная часть помещается в u64 и не превышает 10^l - 1
    const max_decimal = @as(u64, @intFromFloat(scale)) - 1;
    const decimal_u64 = @as(u64, @intCast(decimal));
    if (decimal_u64 > max_decimal) {
        // Эта ситуация не должна возникать, но на всякий случай
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    return .{
        .integer = integer,
        .decimal = decimal_u64,
        .decimal_digits = l,
        .special = false,
        .raw_bits = 0,
    };
}


/// Сжатие массива значений методом Camel.
/// Формат выходных данных:
///   - count (u64, 8 байт) — количество элементов
///   - далее для каждого элемента:
///        если special (NaN, Inf, переполнение):
///            l = 3 (2 бита), flag = 0 (1 бит), raw_bits (64 бита)
///        иначе:
///            сжатая целая часть (алгоритм 1)
///            сжатая дробная часть (алгоритм 2)
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    _ = try configuration.parse(
        allocator,
        configuration.EmptyConfiguration,
        method_configuration,
    );

    var writer = BitWriter.init(allocator, compressed_values);

    const count: u64 = @intCast(uncompressed_values.len);
    try writer.writeBits(count, 64);

    var prev_int: i64 = 0;
    var index: usize = 1;

    for (uncompressed_values) |v| {
        const parts = splitNumber(v, null);

        if (parts.special) {
            try writer.writeBits(3, 2);     // l = 3 (2 bits) as special marker
            try writer.writeBit(0);         // flag = 0
            try writer.writeBits(parts.raw_bits, 64);
        } else {
            try writer.writeBits(parts.decimal_digits, 2);  // l (2 bits)
            try compressInteger(&writer, parts.integer, prev_int, index);
            prev_int = parts.integer;
            try compressDecimalWithoutL(&writer, v, parts.decimal_digits);
        }
        index += 1;
    }

    try writer.finish();
}

// ------------------------------------------------------------
// Algorithm 3: Integer Decompression
// ------------------------------------------------------------
/// Восстанавливает одно целое значение из потока битов.
/// Параметры:
///   - reader: битовый поток для чтения
///   - prev_int: предыдущее восстановленное целое (v_{t-1}.d_int)
///   - index: номер элемента (начиная с 1)
/// Возвращает: v_t.d_int
fn decompressInteger(reader: *BitReader, prev_int: i64, index: usize) Error!i64 {
    if (index == 1) {
        // первое значение: читаем 64 бита как беззнаковое, затем преобразуем в i64
        const raw = reader.readBits(64) catch return Error.CorruptedCompressedData;
        return @as(i64, @bitCast(raw));
    }

    // читаем 2 бита -> range
    const range = reader.readBits(2) catch return Error.CorruptedCompressedData;
    const diff: i64 = blk: {
        if (range != 3) {
            // diff = range - 1 (range 0,1,2 соответствуют diff -1,0,+1)
            break :blk @as(i64, @intCast(range)) - 1;
        } else {
            const symbol = reader.readBit() catch return Error.CorruptedCompressedData; // 1 -> положительный, 0 -> отрицательный
            const flag = reader.readBit() catch return Error.CorruptedCompressedData;   // 0 -> 3 бита, 1 -> 16 бит
            const bits_count = if (flag == 0) @as(u8, 3) else @as(u8, 16);
            var abs_diff = reader.readBits(bits_count) catch return Error.CorruptedCompressedData;
            if (symbol == 0) {
                abs_diff = ~abs_diff + 1; // преобразование в отрицательное
            }
            break :blk @as(i64, @bitCast(abs_diff));
        }
    };
    return prev_int + diff;
}

// ------------------------------------------------------------
// Algorithm 5: Restore (восстановление dxor')
// ------------------------------------------------------------
fn restore(reader: *BitReader, l: u8) Error!u64 {
    if (l <= 1) {
        const bits = l + 1;
        return reader.readBits(bits) catch return Error.CorruptedCompressedData;
    } else if (l == 2) {
        const flag = reader.readBit() catch return Error.CorruptedCompressedData;
        const bits_count = if (flag == 0) @as(u8, 2) else @as(u8, 5);
        return reader.readBits(bits_count) catch return Error.CorruptedCompressedData;
    } else {
        // l >= 3
        const max_val = Math.ceil(Math.log2(Math.pow(f64, 10.0, @as(f64, @floatFromInt(l))) /
                                            Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)))));
        const max = @as(u64, @intFromFloat(max_val));
        const flag = reader.readBits(2) catch return Error.CorruptedCompressedData; // 2 бита → индекс 0..3
        const index = @as(u8, @intCast(flag));
        const bits_to_read = (index + 1) * max / 4;
        return reader.readBits(@as(u8, @intCast(bits_to_read))) catch return Error.CorruptedCompressedData;
    }
}

// ------------------------------------------------------------
// Algorithm 4: Decimal Decompression
// ------------------------------------------------------------
/// Читает флаг и остальные биты согласно алгоритму 4 (без чтения l).
fn decompressDecimalWithL(reader: *BitReader, l: u8) Error!f64 {
    const flag = reader.readBit() catch return Error.CorruptedCompressedData;

    if (flag == 1) {
        const center_bits = reader.readBits(l) catch return Error.CorruptedCompressedData;
        const vd_hat = center_bits << 12;
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_f64 = @as(f64, @floatFromInt(dxor_prime)) / scale;
        const one_plus_dxor = 1.0 + dxor_f64;
        const bits_one_plus = @as(u64, @bitCast(one_plus_dxor));
        const xor_result = bits_one_plus ^ vd_hat;
        const reconstructed = @as(f64, @bitCast(xor_result));
        return reconstructed - 1.0;
    } else {
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        return @as(f64, @floatFromInt(dxor_prime)) / scale;
    }
}


/// Восстанавливает исходный массив значений из сжатых данных Camel.
/// Формат входных данных соответствует формату, создаваемому функцией `compress`.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var reader = BitReader.init(compressed_values);
    const count = reader.readBits(64) catch return Error.CorruptedCompressedData;
    if (count > 1 << 30) return Error.UnsupportedInput;

    var prev_int: i64 = 0;
    var index: usize = 1;

    for (0..@intCast(count)) |_| {
        const l = reader.readBits(2) catch return Error.CorruptedCompressedData;
        const l_u8 = @as(u8, @intCast(l));

        if (l_u8 == 3) {
            const flag = reader.readBit() catch return Error.CorruptedCompressedData;
            if (flag != 0) return Error.UnsupportedInput;
            const raw_bits = reader.readBits(64) catch return Error.CorruptedCompressedData;
            const value = @as(f64, @bitCast(raw_bits));
            try decompressed_values.append(allocator, value);
        } else {
            const int_part = try decompressInteger(&reader, prev_int, index);
            const dec_part = try decompressDecimalWithL(&reader, l_u8);
            const value = @as(f64, @floatFromInt(int_part)) + dec_part;
            try decompressed_values.append(allocator, value);
            prev_int = int_part;
        }
        index += 1;
    }
}



test "camel compresses and decompresses repeated values" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    const distinct_elements: usize = tester.generateBoundRandomInteger(
        usize,
        tester.global_at_least,
        tester.global_at_most,
        null,
    );

    for (0..distinct_elements) |_| {
        const random_value = tester.generateRandomValue(null);
        const repeat: usize = tester.generateBoundRandomInteger(
            usize,
            tester.global_at_least,
            tester.global_at_most,
            null,
        );
        for (0..repeat) |_| {
            try uncompressed_values.append(allocator, random_value);
        }
    }

    const method_configuration = "{}";

    var compressed_values = try tersets.compress(
        allocator,
        uncompressed_values.items,
        Method.Camel,
        method_configuration,
    );
    defer compressed_values.deinit(allocator);

    var decompressed_values = try tersets.decompress(allocator, compressed_values.items);
    defer decompressed_values.deinit(allocator);

    try testing.expect(shared_functions.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        0.0,
    ));
}

test "check camel configuration parsing" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {}
    ;

    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
