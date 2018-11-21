#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "nightly", feature(extern_crate_item_prelude))]

#[cfg(feature = "nightly")]
extern crate test;

use std::ops::{Range, RangeInclusive};

pub trait RangeMin<T> {
    /// find index of smallest element on ```range```
    fn range_min(&self, range: T) -> usize;
}

/// Structure allowing performing range min query on
/// data of i32
/// for which ```data[i] + 1 = data[i+1] || data[i] - 1 == data[i+1]```
/// in O(1) time, O(n) space and O(n) construction speed
///
/// Data split into (n/lg n) chunks of size (n*lg n).
///
/// Hold `RangeMinLn` on minimums of each chunk O(n/lg(n) * ln(n/lg(n))) = O(n) space
/// Which translate query to:
/// + query on right part of leftmost chunk
/// + query on all fully covered chunks | query time = O(1)
/// + query on left part of right most chunk
///
/// since neighboring values can differ only on +1 or -1
/// chunk of size `l` can be represented with string of {+, -}
/// of length `l-1`.
///
/// ```
/// [1, 2, 1, 2, 3, 4, 3, 2, 3, 2, 1]
/// ```
/// will turn into
/// ```
/// + - + + + - - + - -
/// ```
///
/// By replacing every `+` with `0` and every `-` with `1`
/// and writing it as single integer (from least significant bits to most)
/// which is no larger then `2^l = 2^lg(n) = n`
///
/// ##### Note
/// Trailing zeroes do not matter, since adding elements greater then last
/// in the end will not affect neither position nor value of minimal element
///
/// This DS saves offset of minimal element for each such possible chunk
///
/// To perform range min query one needs to get substring of chunk corresponding to
/// specified range and find offset in of min using lookup table.
/// id_of_min = chunk_offset + chunk_range_start + chunk_substring_offset
///
/// This query performed in O(1), therefore full query is also O(1)
pub struct RangeMinLinear {
    data: Vec<i32>,
    /// to perform query on any range of chunks to find which chunk contains minimal value
    chunk_totals: RangeMinLn<i32>,
    /// holds integer representation of chunk
    chunk_content: Vec<usize>,
    /// holds offset of min in each possible chunk and sub_chunk
    // since chunk size is lg(n) offset in chunk cannot be greater then 255
    // as long as size of data does not exceed 2^256
    lookup_table: Vec<u8>,
}

impl RangeMin<Range<usize>> for RangeMinLinear {
    fn range_min(&self, range: Range<usize>) -> usize {
        let chunk_size = self.chunk_size();
        let start_chunk = range.start / chunk_size;
        let end_chunk = (range.end - 1) / chunk_size;

        let s = range.start - start_chunk * chunk_size;
        let e = range.end - end_chunk * chunk_size;
        if start_chunk == end_chunk {
            let bit_string = self.chunk_substring(start_chunk, s..e);
            let offset = self.lookup_table[bit_string];
            return range.start + offset as usize;
        }

        let s = self.chunk_substring(start_chunk, s..chunk_size);
        let s = self.lookup_table[s] as usize + range.start;
        let e = self.chunk_substring(end_chunk, 0..e);
        let e = self.lookup_table[e] as usize + end_chunk * chunk_size;

        if start_chunk + 1 == end_chunk {
            if self.data[s] <= self.data[e] {
                return s;
            } else {
                return e;
            }
        }

        let range = start_chunk + 1..end_chunk;
        let chunk_id = self.chunk_totals.range_min(range);
        let middle =
            chunk_id * chunk_size + self.lookup_table[self.chunk_content[chunk_id]] as usize;

        *[s, middle, e]
            .iter()
            .min_by_key(|&&x| self.data[x])
            .unwrap()
    }
}

impl RangeMinLinear {
    #[inline]
    fn chunk_substring(&self, chunk_id: usize, range: Range<usize>) -> usize {
        let chunk = self.chunk_content[chunk_id];
        (chunk >> range.start) & ((1 << range.end) - 1)
    }

    #[inline]
    fn chunk_size(&self) -> usize {
        lg2(self.data.len())
    }

    pub fn new(data: Vec<i32>) -> Self {
        let n = data.len();
        let chunk_size = lg2(n);
        let lookup_table = Self::build_lookup_table(chunk_size);
        let chunk_content = Self::build_chunk_table(&data, chunk_size);

        let chunk_count = chunk_content.len();
        let mut chunk_mins = vec![0; chunk_count];
        for i in 0..chunk_count {
            let chunk_min_id = i * chunk_size + lookup_table[chunk_content[i]] as usize;
            chunk_mins[i] = data[chunk_min_id];
        }
        let chunk_totals = RangeMinLn::new(chunk_mins);

        Self {
            data,
            chunk_totals,
            chunk_content,
            lookup_table,
        }
    }

    fn build_chunk_table(data: &[i32], chunk_size: usize) -> Vec<usize> {
        let n = data.len();
        let mut chunk_content;
        if n % chunk_size == 0 {
            chunk_content = vec![0; n / chunk_size];
        } else {
            chunk_content = vec![0; n / chunk_size + 1];
            // insert cut chunk
            let chunk = &data[n / chunk_size * chunk_size..];
            let s = Self::chunk_into_bit_string(chunk);
            chunk_content[n / chunk_size] = s;
        }

        for i in 0..n / chunk_size {
            let chunk = &data[i * chunk_size..(i + 1) * chunk_size];
            chunk_content[i] = Self::chunk_into_bit_string(chunk);
        }

        chunk_content
    }

    fn chunk_into_bit_string(chunk: &[i32]) -> usize {
        let mut s = 0;
        for i in 0..chunk.len() - 1 {
            let is_decreasing = chunk[i] > chunk[i + 1];
            s |= (is_decreasing as usize) << i;
        }
        s
    }

    fn build_lookup_table(chunk_size: usize) -> Vec<u8> {
        let lookup_size = 1 << (chunk_size - 1);
        let mut lookup_table = vec![0; lookup_size];
        let mut stack = Vec::with_capacity(chunk_size - 1);
        lookup_table[1] = 1;
        stack.push((1usize, -1i32, 1u8, 1));
        while let Some((chunk, min, min_id, depth)) = stack.pop() {
            if depth < chunk_size - 1 {
                // calc for string (+, chunk)
                let s0 = chunk << 1;
                let (s0_min, s0_min_id) = if min >= 0 {
                    (0, 0)
                } else {
                    (min + 1, min_id + 1)
                };
                stack.push((s0, s0_min, s0_min_id, depth + 1));
                lookup_table[s0] = s0_min_id;

                // calc for string (-, chunk)
                let s1 = s0 | 1;
                let s1_min = min - 1;
                let s1_min_id = min_id + 1;
                stack.push((s1, s1_min, s1_min_id, depth + 1));
                lookup_table[s1] = s1_min_id;
            }
        }
        lookup_table
    }
}

/// Structure allowing performing range min query in O(1) time
/// with O(n*ln n) space and O(n*ln n) construction speed
struct RangeMinLn<T> {
    data: Vec<T>,
    /// chunk_min[i][j] holds id of smallest element on `data[j..j+(1<<(i + 1))]`
    chunk_min: Vec<Vec<usize>>,
}

impl<T: Ord> RangeMinLn<T> {
    /// create new instance of RangeMinLn
    ///
    /// panic if data.empty()
    fn new(data: Vec<T>) -> Self {
        let n = data.len();
        let layer_count = lg2(n) - 1;
        let mut chunk_min;

        chunk_min = Vec::with_capacity(layer_count);

        // construct first layer (for chunks of size 2)
        let mut layer = vec![0; n - 1];
        for (i, v) in layer.iter_mut().enumerate() {
            if data[i] <= data[i + 1] {
                *v = i;
            } else {
                *v = i + 1;
            }
        }
        chunk_min.push(layer);

        for i in 1..layer_count {
            let layer_size = 1 << (i + 1);
            let mut layer = vec![0; n + 1 - layer_size];
            let prev = &chunk_min[i - 1];
            for (j, v) in layer.iter_mut().enumerate() {
                let l = prev[j];
                let r = prev[j + layer_size / 2];
                if data[l] <= data[r] {
                    *v = l;
                } else {
                    *v = r;
                }
            }
            chunk_min.push(layer);
        }

        Self { data, chunk_min }
    }
}

impl<T: Ord> RangeMin<Range<usize>> for RangeMinLn<T> {
    fn range_min(&self, range: Range<usize>) -> usize {
        let len = range.end - range.start;
        if len == 1 {
            range.start
        } else {
            let chunk_id = lg2(len) - 2;
            let chunk_len = 1 << (chunk_id + 1);
            let chunks = &self.chunk_min[chunk_id];
            let l = chunks[range.start];
            let r = chunks[range.end - chunk_len];
            if self.data[l] <= self.data[r] {
                l
            } else {
                r
            }
        }
    }
}

impl<T: Ord> RangeMin<RangeInclusive<usize>> for RangeMinLn<T> {
    #[allow(clippy::range_plus_one)]
    fn range_min(&self, range: RangeInclusive<usize>) -> usize {
        let (s, e) = range.into_inner();
        self.range_min(s..e + 1)
    }
}

fn lg2(mut x: usize) -> usize {
    let mut i = 0;
    while x > 0 {
        i += 1;
        x /= 2;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_new() {
        let data = vec![1, 2, 3, 2, 1, 0, -1, -2, -1, 0, -1];
        let rml = RangeMinLinear::new(data);

        assert_eq!(rml.chunk_totals.data, [1, -2, -1]);
        assert_eq!(rml.lookup_table[rml.chunk_content[0]], 0);
        assert_eq!(rml.lookup_table[rml.chunk_content[1]], 3);
        assert_eq!(rml.lookup_table[rml.chunk_content[2]], 2);
    }

    #[test]
    fn test_build_chunk_table() {
        let data = &[1, 2, 3, 2, 1, 0, -1, -2, -1, 0, -1];
        let chunk_size = 3;
        let expected = &[0b00, 0b11, 0b01, 0b1];
        assert_eq!(
            RangeMinLinear::build_chunk_table(data, chunk_size),
            expected
        );

        let data = &[1, 2, 3, 2, 1, 0, -1, -2, -1];
        let chunk_size = 3;
        let expected = &[0b00, 0b11, 0b01];
        assert_eq!(
            RangeMinLinear::build_chunk_table(data, chunk_size),
            expected
        );
    }

    #[test]
    fn test_chunk_to_bit_string() {
        let chunk = &[1, 2, 3, 2, 1, 0, -1, -2, -1, 0, -1];
        // written in reverse
        let expected = 0b10_0111_1100;
        assert_eq!(expected, RangeMinLinear::chunk_into_bit_string(chunk));
    }

    #[test]
    fn test_build_lookup() {
        let table = RangeMinLinear::build_lookup_table(10);
        for (i, value) in table.iter().enumerate() {
            let mut acc = 0i32;
            let (min_id, _min) = (0..16)
                .map(|k| (i >> k) & 1)
                .map(|b| {
                    let old = acc;
                    acc += 1 - 2 * b as i32;
                    old
                })
                .enumerate()
                //.inspect(|&(j, acc)| println!("chunk = {}, i = {}, acc = {}", i, j, acc))
                .min_by_key(|&(i, acc)| (acc, -(i as i32)))
                .unwrap();
            assert_eq!(min_id, *value as usize, "error on value = 0b{:b}", i);
        }
        assert_eq!(table.len(), 512);
    }

    #[test]
    fn test_range_min_ln_create() {
        let mut data = vec![0; 10];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i % 5;
        }

        let c = RangeMinLn::new(data);
        assert_eq!(c.chunk_min[0], vec![0, 1, 2, 3, 5, 5, 6, 7, 8]);
        assert_eq!(c.chunk_min[1], vec![0, 1, 5, 5, 5, 5, 6]);
    }

    #[test]
    fn test_range_min_ln_query() {
        let mut data = vec![0; 10];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i % 5;
        }
        let c = RangeMinLn::new(data);

        assert_eq!(c.range_min(2..7), 5);
    }

    #[cfg(feature = "nightly")]
    use test::Bencher;

    macro_rules! prepare_data {
        ($size: expr) => {{
            let mut data = vec![0i32; $size];
            for i in 0..$size {
                data[i] = i as i32 % 10;
                if data[i] > 5 {
                    data[i] = 10 - data[i];
                }
            }
            data
        }};
    }

    #[test]
    fn test_query1() {
        let data = prepare_data!(100);
        let ds = RangeMinLn::new(data);
        let id = ds.range_min(6..80);
        assert_eq!(id, 10);
    }

    #[test]
    fn test_query2() {
        let data = prepare_data!(100);
        let ds = RangeMinLinear::new(data);
        let id = ds.range_min(6..80);
        assert_eq!(id, 10);
    }

    macro_rules! bench_new {
        ($name: ident, $ds: ident, $size: expr) => {
            #[cfg(feature = "nightly")]
            #[bench]
            fn $name(b: &mut Bencher) {
                let data = prepare_data!($size);
                b.iter(|| {
                    let _c = $ds::new(data.clone());
                })
            }
        };
    }

    macro_rules! bench_range_min {
        ($name: ident, $ds: ident, $size: expr) => {
            #[cfg(feature = "nightly")]
            #[bench]
            fn $name(b: &mut Bencher) {
                let data = prepare_data!($size);
                let ds = $ds::new(data.clone());
                b.iter(|| {
                    let _c = ds.range_min($size / 3..2 * $size / 3);
                })
            }
        };
    }

    bench_new!{rml_new_100, RangeMinLn, 100}
    bench_new!{rml_new_100_000, RangeMinLn, 100_000}
    bench_new!{rmlin_new_100, RangeMinLinear, 100}
    bench_new!{rmlin_new_100_000, RangeMinLinear, 100_000}

    bench_range_min!{rml_range_min_100, RangeMinLn, 100}
    bench_range_min!{rml_range_min_100_000, RangeMinLn, 100_000}
    bench_range_min!{rmlin_range_min_100, RangeMinLinear, 100}
    bench_range_min!{rmlin_range_min_100_000, RangeMinLinear, 100_000}
}
