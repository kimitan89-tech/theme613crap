//! Samplers as the host describes them, from the console's sampler
//! description (`CreateSamplerX`), which names the hardware's sampler
//! fields. (Immediate samplers in shader code come decoded with the
//! translation.)

use windows::Win32::Graphics::Direct3D12::*;

/// The fields both forms share.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fields {
    /// Point 0, bilinear 1, anisotropic point 2, anisotropic bilinear 3.
    mag: u32,
    min: u32,
    /// None 0, point 1, linear 2.
    mip: u32,
    /// Plain 0, minimum 1, maximum 2.
    mode: u32,
    /// Host address modes.
    address: [u32; 3],
    anisotropy: u32,
    /// Host comparison function.
    compare: u32,
    min_lod: f32,
    max_lod: f32,
    bias: f32,
    border: [f32; 4],
}

/// Optional anisotropic filtering floor (`VICTORY_ANISO=1..16`; unset leaves
/// title samplers unchanged). Only upgrades mipmapped linear/aniso samplers so
/// point-sampled G-buffers, LUTs, and UI stay exact.
fn aniso_override() -> Option<u32> {
    static ANISO: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *ANISO.get_or_init(|| {
        std::env::var("VICTORY_ANISO")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&a| a >= 1)
            .map(|a| a.min(16))
    })
}

/// Optional mip LOD bias offset (`VICTORY_MIP_BIAS`, e.g. `-0.5` for sharper
/// distant mips, clamped to `-4.0..=4.0`; default `0.0`).
fn mip_bias_override() -> f32 {
    static BIAS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *BIAS.get_or_init(|| {
        std::env::var("VICTORY_MIP_BIAS")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map_or(0.0, |v| v.clamp(-4.0, 4.0))
    })
}

impl Fields {
    fn host(&self) -> D3D12_SAMPLER_DESC {
        self.host_with_overrides(aniso_override(), mip_bias_override())
    }

    fn host_with_overrides(&self, aniso_floor: Option<u32>, mip_bias_delta: f32) -> D3D12_SAMPLER_DESC {
        let linear = |f: u32| u32::from(f == 1 || f == 3);
        let mipmapped_linear = self.mag >= 1 && self.min >= 1 && self.mip >= 1;
        let aniso = self.mag >= 2 || self.min >= 2 || (aniso_floor.is_some() && mipmapped_linear);
        // The console compares when the instruction says so, with the
        // sampler's function; the host needs a comparison sampler for that.
        // A sampler given a real test (less ... greater-or-equal) is one;
        // never, always and none are plain samplers.
        const LESS: u32 = 2;
        const GREATER_EQUAL: u32 = 7;
        let compares = (LESS..=GREATER_EQUAL).contains(&self.compare);
        let reduction = match self.mode {
            1 => 2,
            2 => 3,
            _ => u32::from(compares),
        };
        let filter = if aniso {
            0x55
        } else {
            linear(self.min) << 4 | linear(self.mag) << 2 | u32::from(self.mip == 2)
        };
        let max_aniso = if aniso {
            let base = self.anisotropy.clamp(1, 16);
            aniso_floor.map_or(base, |floor| base.max(floor).clamp(1, 16))
        } else {
            1
        };
        let bias = if mipmapped_linear {
            (self.bias + mip_bias_delta).clamp(-16.0, 15.99)
        } else {
            self.bias
        };
        D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER((reduction << 7 | filter) as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(self.address[0] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(self.address[1] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(self.address[2] as i32),
            MipLODBias: bias,
            MaxAnisotropy: max_aniso,
            ComparisonFunc: D3D12_COMPARISON_FUNC(self.compare.max(1) as i32),
            BorderColor: self.border,
            MinLOD: self.min_lod,
            // Without a mip filter the console samples mip 0 only.
            MaxLOD: if self.mip == 0 {
                self.min_lod
            } else {
                self.max_lod
            },
        }
    }
}

/// The console's sampler description (27 words): flags, magnification,
/// minification, mip and depth filters, filter mode, three address modes,
/// minimum and maximum LOD, LOD bias, secondary bias, anisotropy, its bias
/// and threshold, comparison, border color, then tuning words.
pub(crate) fn console(words: &[u32; 27]) -> D3D12_SAMPLER_DESC {
    let f = |i: usize| f32::from_bits(words[i]);
    Fields {
        mag: words[1],
        min: words[2],
        mip: words[3],
        mode: words[5],
        address: [words[6], words[7], words[8]],
        min_lod: f(9),
        max_lod: f(10),
        bias: f(11),
        anisotropy: words[13],
        compare: words[16],
        border: [f(17), f(18), f(19), f(20)],
    }
    .host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_descriptions_decode() {
        let mut w = [0u32; 27];
        w[1] = 1; // bilinear magnification
        w[2] = 0; // point minification
        w[3] = 2; // linear mips
        w[6..9].copy_from_slice(&[3, 3, 3]);
        w[10] = 1000.0f32.to_bits();
        let d = console(&w);
        assert_eq!(d.Filter.0, 0x5, "min point, mag and mip linear");
        assert_eq!(d.AddressW, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(d.MaxLOD, 1000.0);
        // A less-or-equal test makes a comparison sampler; never does not.
        w[16] = 4;
        assert_eq!(console(&w).Filter.0, 0x85);
        assert_eq!(console(&w).ComparisonFunc, D3D12_COMPARISON_FUNC_LESS_EQUAL);
        w[16] = 1;
        assert_eq!(console(&w).Filter.0, 0x5);
    }

    #[test]
    fn overrides_upgrade_mipmapped_linear_and_leave_point_samplers_exact() {
        let linear_mip = Fields {
            mag: 1,
            min: 1,
            mip: 2,
            mode: 0,
            address: [1, 1, 1],
            anisotropy: 4,
            compare: 1,
            min_lod: 0.0,
            max_lod: 1000.0,
            bias: 0.0,
            border: [0.0; 4],
        };
        let upgraded = linear_mip.host_with_overrides(Some(16), -0.5);
        assert_eq!(upgraded.Filter.0, 0x55, "upgraded to anisotropic filter");
        assert_eq!(upgraded.MaxAnisotropy, 16);
        assert_eq!(upgraded.MipLODBias, -0.5);

        let point = Fields { mag: 0, min: 0, ..linear_mip };
        let unchanged = point.host_with_overrides(Some(16), -0.5);
        assert_ne!(unchanged.Filter.0, 0x55, "point sampler stays point");
        assert_eq!(unchanged.MaxAnisotropy, 1);
        assert_eq!(unchanged.MipLODBias, 0.0);
    }
}
