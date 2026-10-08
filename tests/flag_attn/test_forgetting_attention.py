"""Self-contained Forgetting Attention / ACP correctness tests.

Run: python -m pytest tests/flag_attn/test_forgetting_attention.py -v
No JSON configuration, custom pytest markers or test-support package is needed.
The benchmark imports the same inputs and reference adapters from this file.
"""
import gc
import hashlib
import importlib
import weakref
from functools import lru_cache

import pytest
import torch
import torch.nn.functional as F

from flag_attn.forgetting_attention import has_tle
from flag_attn.forgetting_attention.naive import OFFICIAL_COMMIT

# Inline pytest parameters: B, M, N, Hq, Hkv, D, dtype, scale, seed-0 SHA256.
# Preserve all 90 historical configurations and their numerical regression oracle.
_CASE_ROWS = [
    (1, 63, 63, 1, 1, 64, 'float16', 1.0, '84fd21a08105551e1dfab4a71c8ff2da9c95ed07dcc2598e9cbdb4315cfa6c24'),
    (3, 111, 111, 2, 2, 100, 'float16', 1.0, '51cc03a3fd79785e209475a5d2fccf39cbd2087a2c4918ab0a59104e6586af6b'),
    (3, 1024, 1024, 8, 2, 60, 'float16', 0.1, 'e984f8893033fc62719da27b3e0c66c5ad1057890d94159d1f5cbb985cedb8cd'),
    (3, 1024, 1024, 8, 2, 128, 'float16', 0.1, 'd54409fd2540c8f705419a929d16c5a4c0e02bbdf83b5dcefe0fe8ab89b14a16'),
    (4, 2048, 2048, 8, 2, 64, 'float16', 0.1, 'f254fb73c4b59ded3f9dbe87532cb566e9ce3d3d87863adc052bce9cf8cb3507'),
    (4, 1024, 1024, 32, 32, 64, 'bfloat16', 1.3, '16d41f5f98f8a8b60aa231b565b9a168d6a766548f66662d4a563e2b2c259dfc'),
    (4, 2048, 2048, 32, 32, 64, 'bfloat16', 1.3, 'e9a0b05c6c9a158582a480287f25050d37c6a659bc6dcdb3ca1cbae09765dec1'),
    (4, 2432, 2432, 32, 32, 64, 'bfloat16', 1.3, '519b8eca3bd48b91436f6062951e933dd3f53ced900090da8c362090970384ef'),
    (4, 3008, 3008, 32, 32, 64, 'bfloat16', 1.3, '6681c478c9766712f8277fc49c04d570d7423199140e69ed0c9adafb4ee96768'),
    (4, 3072, 3072, 32, 32, 64, 'bfloat16', 1.3, '73758a8272cc3a44af90b02301ad705082fab875d16d13dfc3b2918301d11da8'),
    (4, 4096, 4096, 32, 32, 64, 'bfloat16', 1.3, '6e130f3fc8e398894909f32cf5df81510662fdef0c43fdf6919ae03beaf54d9f'),
    (4, 4160, 4160, 32, 32, 64, 'bfloat16', 1.3, 'd0fb51d3351f2ff76eae758ca76583d7e9b07f2551b942ed77d3621f767229f0'),
    (4, 6144, 6144, 32, 32, 64, 'bfloat16', 1.3, '49ec60f3f4e9e136032165ccfc74a80cd161f1dac07ebac59c9b49bd718257e6'),
    (4, 8192, 8192, 32, 32, 64, 'bfloat16', 1.3, '4f80ab51a4379dd3d39fff062f170ab5ad429d5c02800741a773449c6220fb72'),
    (4, 2368, 2368, 32, 32, 64, 'bfloat16', 1.3, '9b8db928a1688b4c2bfc3020ab25ad74712ff126a32c925af894bc963ecf69f3'),
    (4, 8256, 8256, 32, 32, 64, 'bfloat16', 1.3, '70027be79fc32a3e3e16826d276bb40ca697fec0acdf94c3127c37032de5ff33'),
    (1, 4096, 4096, 1, 1, 64, 'bfloat16', 1.3, '9a126c50caafd49ac69212096f14adceb436e5b63bcc466e29ba928750b32e79'),
    (1, 4096, 4096, 8, 8, 64, 'bfloat16', 1.3, '1b7d11f2c3cf0ce9ef0f2829fe2433915cbc82182b79e0e38d0124c808c2e04d'),
    (1, 4096, 4096, 32, 32, 64, 'bfloat16', 1.3, 'dcf6571a03d8eb7b7f927ef6daeb29903529826c101def71026116aff78f06d2'),
    (2, 4096, 4096, 8, 8, 64, 'bfloat16', 1.3, 'ab397d7200752df45c35e7587c93fa80ad0af13729da861dec851fcd8b770c05'),
    (4, 4096, 4096, 8, 8, 64, 'bfloat16', 1.3, 'de40c3514e08b56204295df55f95f30caf970ff90f565ada159751931f9e10f4'),
    (8, 4096, 4096, 8, 8, 64, 'bfloat16', 1.3, 'a813d728808e8eb619503ac7155d751d5414c208f6bd530bcaeb09ef9ec24b7b'),
    (4, 4096, 4096, 32, 32, 32, 'bfloat16', 1.3, 'e44504132c4842889339ae2ff4b03191bdeb3e2c94ff208b56cd9f75adbcf442'),
    (4, 4096, 4096, 32, 32, 128, 'bfloat16', 1.3, '8c8c7f771d16134a553e5e97d4d55876e817eb6394794adc1a380c74d968599a'),
    (2, 4096, 4096, 32, 32, 64, 'bfloat16', 1.3, 'a3b94dfda11c2574bd3fe48173485c970afb5ab9267fb384be291851f6fd6efb'),
    (8, 4096, 4096, 32, 32, 64, 'bfloat16', 1.3, 'e6cdafa504f7b2158ba89e459ba7eb9c6d790f5e9200d3b977dfe2cd0dd54a0b'),
    (4, 4096, 4096, 16, 16, 64, 'bfloat16', 1.3, 'dfded2fcd1e7a04c17e402ee8138136a3b47b80073c53d795b99b1b0ec9ad147'),
    (4, 4096, 4096, 64, 64, 64, 'bfloat16', 1.3, '28421166d28f0fd5283ccd94f0cf4efe5f453cafc0f1bfe85b6c11aaa28abad3'),
    (2, 4096, 4096, 16, 16, 64, 'bfloat16', 1.3, '4a7a0bf1e268dabef6f424637ffb6d723195692e69445d6ce75d80ac3468e3ac'),
    (2, 4096, 4096, 64, 64, 64, 'bfloat16', 1.3, '5dc93d12e02927a49b6b611346ac732558ef6429cff2120dbfee8198dbaad80d'),
    (2, 1, 1, 8, 8, 64, 'bfloat16', 1.3, '5016f991298a8c84667331a17da8b0272cc58b42ef856ce5e0ee3b65439b3770'),
    (2, 63, 63, 8, 8, 64, 'bfloat16', 1.3, '99d706b5b6900735e954de754098b692de288740638020f41c4c83c9e0197d51'),
    (2, 64, 64, 8, 8, 64, 'bfloat16', 1.3, '9033fbc7cc4e5acc3c82cca6194d01498b9bd2767126f4d33632d84f85543473'),
    (2, 65, 65, 8, 8, 64, 'bfloat16', 1.3, 'f9e0b977ed7e0e9533a2478ad5a969b946dac0318352ce3a3745a97ddc4e0062'),
    (2, 127, 127, 8, 8, 64, 'bfloat16', 1.3, 'b433f54ae49924dd2a68659841466185a490298672d26d114731b118f252c6b8'),
    (2, 128, 128, 8, 8, 64, 'bfloat16', 1.3, '013cfd6a217830d0aac9de05a70acf940828723606d1adb0426f646db90e7ced'),
    (2, 129, 129, 8, 8, 64, 'bfloat16', 1.3, '72bc0bb70735e7159ec0608648d5ae34175316ec8f8eee502ffce84b1653c301'),
    (2, 1023, 1023, 8, 8, 64, 'bfloat16', 1.3, 'c83e8f76ab60fa4a1d1cc989127bf1ebb8a34ada03439455f72e62813be07ba1'),
    (2, 1024, 1024, 8, 8, 64, 'bfloat16', 1.3, '6beea218dcf4cf0db83bba809846f634a5d8dac24edba8ecd5a0a5f0fa2fa1b0'),
    (2, 1025, 1025, 8, 8, 64, 'bfloat16', 1.3, '9ccafb1d5d36e5d16c0fceb353a94bebd2777f0c36feb1ce6075af7c5ca8d245'),
    (2, 2047, 2047, 8, 8, 64, 'bfloat16', 1.3, '2304fb9e4d7adc603b5d9256d9ca2cf932c9527bf0b8da8e9db8c50bad352a97'),
    (2, 2048, 2048, 8, 8, 64, 'bfloat16', 1.3, 'c385eba23257febd10a5878231d1373f335a5080e02f78b93afffcdae6de3b35'),
    (2, 2049, 2049, 8, 8, 64, 'bfloat16', 1.3, '6567bf0cbd4aef3fce79bd79a02f7911a8aecb8646ac31ffd9511c6401e0844c'),
    (2, 4095, 4095, 8, 8, 64, 'bfloat16', 1.3, '4d4b8701b8fea033941dd70b322e9473b44d9b849cb9fe5ae9482489654c88dc'),
    (2, 4097, 4097, 8, 8, 64, 'bfloat16', 1.3, 'aacd421e318b4fcd9ad4ea4757f89271ddbf1b86b76eb967bc34b9a1dd73da3d'),
    (2, 8191, 8191, 8, 8, 64, 'bfloat16', 1.3, '799a4699afd26a28ef9696961e3225a2b3d934b178acc9f60a899436614dcc63'),
    (2, 8192, 8192, 8, 8, 64, 'bfloat16', 1.3, '68e671b8213a9640aa26067f38d1002848cf485002e8fc4267b9f3f9333f45e5'),
    (2, 8193, 8193, 8, 8, 64, 'bfloat16', 1.3, '42a99994d03bf89715e2c8c282a19a4d48e31aa9dc606ac4ffd7b0c62ae75f5a'),
    (2, 1025, 1025, 8, 8, 16, 'float16', 1.3, '30a4a624bb9f44dfb64d1355d81b3a8df7ca7cc77123d565f402b014ca1498ac'),
    (2, 1025, 1025, 8, 8, 32, 'float16', 1.3, '69104f4e9de77f5aa307a2db989f2052ecf436c47f79c614a373409bbdc0a5ad'),
    (2, 1025, 1025, 8, 8, 64, 'float16', 1.3, '92fba2f68bf2068ad40b54e40b564fcf0d5f2d0307eb293c8e7e88072410dc2e'),
    (2, 1025, 1025, 8, 8, 128, 'float16', 1.3, 'e1d6801ceba2092c4be43d69ff0afcbbb05f2a922f7f7bbeb3174d45ffde2b1d'),
    (2, 1025, 1025, 8, 8, 16, 'bfloat16', 1.3, '5f55781325a02397e40caf0268ef1b16c55f298b7af32d3e8fe5049dcc3dd339'),
    (2, 1025, 1025, 8, 8, 32, 'bfloat16', 1.3, '61a18d5f651df4c316b9fc8ce9d9e7ad9558941fd1ea2fe82018a2781be13998'),
    (2, 1025, 1025, 8, 8, 128, 'bfloat16', 1.3, 'f772f6b45f46ab92fe65a23997d34b050b1f35a471bef160ffeaeddd1bbbfd70'),
    (1, 2048, 2048, 1, 1, 64, 'bfloat16', 1.3, 'c1e129b5d438f3ae112de09b5cb94dbf445b22dc60dd6443ee77273ce86cf7af'),
    (1, 2048, 2048, 8, 8, 64, 'bfloat16', 1.3, 'e3b031ea3b4c0e2ed7332f537241664ad2e32b9c91b7d74fb872ccef8ded95d8'),
    (1, 2048, 2048, 32, 32, 64, 'bfloat16', 1.3, 'a76c00886aa8d35b7458788bee7c3bc0239641ead0d6b478e96fc2ef67edbc68'),
    (8, 2048, 2048, 8, 8, 64, 'bfloat16', 1.3, '4b459a96132271ca99dca545fb9a5a47c90dbf423310616935cbc0962875d0b6'),
    (2, 2048, 2048, 64, 64, 64, 'bfloat16', 1.3, '2938b2635778f983c65094521f3800dbc82108c22373354a1286840e59fe0935'),
    (8, 2048, 2048, 32, 32, 64, 'bfloat16', 1.3, '9ab9f06c6531a5311760075f62aaca4ef121c43b9d6605e34952c913a251c50e'),
    (2, 1, 63, 8, 8, 64, 'bfloat16', 1.3, 'b5aa15a9fbc681ee38ea481c9fbda979156617d1df1999a02c42eed97fc74ee9'),
    (2, 63, 111, 8, 8, 64, 'bfloat16', 1.3, 'd6c41bcdc6c85049635ab0742a3ac050b25c7b9bc562d21933ecd060cd044d1c'),
    (4, 1020, 2098, 2, 2, 64, 'bfloat16', 1.3, 'f5bf3031a23fa49cd2c0f7d9b6369451d8c24479e98cd5298aef5baa00644ea0'),
    (4, 1024, 2048, 2, 2, 64, 'bfloat16', 1.3, '8396c211afea0718bed4fd648fed8c98a13e239bb74c596ba024c256ddd15152'),
    (2, 2048, 4096, 8, 8, 64, 'bfloat16', 1.3, '1fd206990a288cffc8bef000866433a309b6d1cbc90794b9f657358e7bcb5b41'),
    (2, 4096, 8192, 8, 8, 64, 'bfloat16', 1.3, '5a701be574af1ad1a7dba560cc54a942a1c0f4439ad1e05e1a019763ae0d9875'),
    (4, 2048, 2048, 8, 8, 64, 'bfloat16', 1.3, 'eff879ee9d25ee036d8b569fb304d6a4e28435d6b4c7578d5af357d74307da1e'),
    (2, 257, 257, 8, 8, 64, 'bfloat16', 1.3, '91ff9f97f584b6de0c737059cf8efc68bef2fb00bd25a63c5492bef667ad503a'),
    (1, 63, 63, 1, 1, 64, 'bfloat16', 1.3, 'b14654d5a2d1c3078ee3fc0e53589de0335ab06b055dc575efe7e7478a1307c5'),
    (2, 257, 257, 8, 8, 64, 'float16', 1.3, 'a2f067df8da8c236d59ad2e6810f1e3e6b366b46c4a60e863d3066a0ee344af9'),
    (2, 1024, 1024, 8, 8, 128, 'bfloat16', 1.3, 'a6a7038840f05df355651d982a5f921a6940474bac86119dab10e245b8b293bb'),
    (2, 512, 1024, 8, 8, 64, 'bfloat16', 1.3, 'd0dc014a8353e697266b6cc9bc1ec61af8df395edb9af611f29f59cea248aba7'),
    (4, 2496, 2496, 32, 32, 64, 'bfloat16', 1.3, '1e24b4131db2d6dcf601fed1cffaa5768a7a19df1c188087b1bc30b7df02ecb8'),
    (4, 2560, 2560, 32, 32, 64, 'bfloat16', 1.3, '21eef29a0e58501e33de750d7c3b3bd8ee1d266dcede19300d9d19f779a61752'),
    (4, 2624, 2624, 32, 32, 64, 'bfloat16', 1.3, '19af21dc27e73501c619fbafe9df577b11b6e42ba49e61087aaae59423c3688d'),
    (4, 2688, 2688, 32, 32, 64, 'bfloat16', 1.3, '148625cee228c32e01ed7768879cccbb13a6f6b6c9913ce9b4e653f5d2891078'),
    (4, 2752, 2752, 32, 32, 64, 'bfloat16', 1.3, '4af267aeff63372a421af8ae62af645aa26ffcc4d0d9c16799325fdf93800c15'),
    (4, 2816, 2816, 32, 32, 64, 'bfloat16', 1.3, 'e59b78caa6d421ecfb6925f00bd2118cc149ec30322c72886d0fd0b42b747dff'),
    (4, 2880, 2880, 32, 32, 64, 'bfloat16', 1.3, '70362c7b4131df17c24ed585f5bcc6cdb4cd0d3809dadc5b04bc787e7b008205'),
    (4, 2944, 2944, 32, 32, 64, 'bfloat16', 1.3, '942bdbc2df91a0b2de79256bc216184e7ecb87c70a715c858ddfcb91c3a909d4'),
    (4, 2559, 2559, 32, 32, 64, 'bfloat16', 1.3, '7ec6fd7d9267654f935ad34537af17ff052ca89cbd66e654441268109a117057'),
    (4, 2561, 2561, 32, 32, 64, 'bfloat16', 1.3, '8b87eca7c0170d23b9d9ca899ae3cb509d60db3a568504fab836e58667ed21d6'),
    (4, 2623, 2623, 32, 32, 64, 'bfloat16', 1.3, '53d19ec461b6d9a3c15b480c4cd7255cf80456f1b445d38e739b7828c96270e0'),
    (4, 2625, 2625, 32, 32, 64, 'bfloat16', 1.3, '94ee98c02cd80db5c164263c63c096bb7390db076fa638ac08609ab5e079d303'),
    (4, 2687, 2687, 32, 32, 64, 'bfloat16', 1.3, '689660123e256698de4a0c054172e828ebbb88db2ba59993d226ae7dbda11a7c'),
    (4, 2689, 2689, 32, 32, 64, 'bfloat16', 1.3, '128381b257cc859b155a20cdd74b086683eb128a263a880d79f559898ebaed8e'),
    (4, 1020, 2098, 2, 2, 64, 'float16', 1.3, 'e2c6da1032c25867ac0ff25df8acd62934c5df56ffae59e70586ab80d5deeff5'),
    (4, 1024, 2048, 2, 2, 128, 'float16', 1.3, '1ec91df4c810db80a679eb11dec81814932cb0980fb890d5c3bc236007054c63'),
    (4, 4096, 4096, 32, 32, 64, 'float16', 1.3, '6846909f696d662ee951c7e1676ec29f66579867b186bfb6fd33c5cc83c37282'),
]
_CASE_FIELDS = ("B", "M", "N", "HQ", "H", "D", "dtype", "scale", "v66_frozen_sha256")
CASES = []
for _index, _row in enumerate(_CASE_ROWS, 1):
    _case = dict(zip(_CASE_FIELDS, _row))
    _case["id"] = f"S{_index:03d}"
    _case["reference_kind"] = (
        "native" if _case["D"] in (16, 32, 64, 128) and _case["HQ"] == _case["H"] else "adapted"
    )
    CASES.append(_case)
del _index, _row, _case
DEFAULT_IDS = ("S001", "S003", "S011", "S024", "S031", "S034", "S064")
DEFAULT_CASES = [case for case in CASES if case["id"] in DEFAULT_IDS]

def case_id(case):
    return f'{case["id"]}-{case["dtype"]}-{case["reference_kind"]}'


def make_inputs(case, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    b, m, n, hq, hkv, d = (case[k] for k in ("B", "M", "N", "HQ", "H", "D"))
    dtype = getattr(torch, case["dtype"])
    q = torch.randn((b, m, hq, d), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((b, n, hkv, d), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn((b, n, hkv, d), device="cuda", dtype=dtype, generator=generator)
    gate = F.logsigmoid(torch.empty((b, n, hq), device="cuda", dtype=torch.float32)
                       .uniform_(0.0, 10.0, generator=generator))
    return q, k, v, gate


@lru_cache(maxsize=1)
def optimized_entry():
    return importlib.import_module("flag_attn.forgetting_attention").forgetting_attention


@lru_cache(maxsize=1)
def official_entry():
    return importlib.import_module("flag_attn.forgetting_attention.naive").forgetting_attention


def call_optimized(inputs, case, threshold=-10.0, scale=None):
    return optimized_entry()(*inputs, head_first=False, seq_start=None,
                             sm_scale=case["scale"] if scale is None else scale,
                             adaptive_threshold=threshold)


def call_official(inputs, case, threshold=-10.0, scale=None):
    q, k, v, gate = inputs
    # Explicit adapter, NOT a claim that upstream natively supports these cases.
    # All padding/repetition/cropping remains inside the timed reference call.
    d = case["D"]
    padded = 1 << (d - 1).bit_length()
    if d != padded:
        q, k, v = (F.pad(t, (0, padded - d)) for t in (q, k, v))
    groups = case["HQ"] // case["H"]
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=2) for t in (k, v))
    out = official_entry()(q, k, v, gate, head_first=False, seq_start=None,
                           sm_scale=case["scale"] if scale is None else scale,
                           adaptive_threshold=threshold)
    return out if d == padded else out[..., :d].contiguous()


def fingerprint(tensor):
    data = tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all().item(), "non-finite optimized output"
    assert torch.isfinite(expected).all().item(), "non-finite reference output"
    assert torch.equal(actual.contiguous().view(torch.int16),
                       expected.contiguous().view(torch.int16)), (
        f"outputs differ; max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


CUDA_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = [
    pytest.mark.skipif(not CUDA_SM90, reason="ACP V7.6 requires Hopper SM90 CUDA"),
    pytest.mark.skipif(not has_tle(), reason="ACP V7.6 requires compatible Triton 3.6 TLE"),
]


@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_forgetting_attention_matches_official(case, seed):
    inputs = make_inputs(case, seed)
    expected = call_official(inputs, case)
    cold = call_optimized(inputs, case)
    warm = call_optimized(inputs, case)
    assert_bitwise(cold, expected)
    assert_bitwise(warm, expected)
    if seed == 0 and torch.__version__.startswith("2.10."):
        assert fingerprint(warm) == case["v66_frozen_sha256"], "V7.6 frozen-output regression"


@pytest.mark.parametrize("case", [c for c in CASES if c["id"] in
                                  ("S001", "S003", "S011", "S024", "S031", "S034", "S064")], ids=case_id)
@torch.inference_mode()
def test_forgetting_attention_prefix_and_boundaries(case):
    from flag_attn.forgetting_attention.parallel import prepare, h100_config
    inputs = make_inputs(case)
    b, m, n, h, hk, d = (case[k] for k in ("B", "M", "N", "HQ", "H", "D"))
    prep = h100_config(b, m, n, h, hk, d)[2]
    prefix, starts, _, _, bn = prepare(*inputs, False, None, case["scale"], -10.0, prep, True)
    expected_prefix = torch.cumsum(inputs[3].transpose(1, 2), -1, dtype=torch.float32)
    assert torch.equal(prefix, expected_prefix)
    qm = 1 if m == 1 else 128
    qpos = n - m + torch.arange(0, m, qm, device="cuda")
    kpos = torch.arange(bn - 1, n + bn - 1, bn, device="cuda").clamp_max(n - 1)
    anchors = expected_prefix[..., qpos]
    ends = expected_prefix[..., kpos]
    expected_starts = ((anchors[..., :, None] - ends[..., None, :]) < -10.0).sum(-1) * bn
    assert torch.equal(starts, expected_starts)


def _reuse_case(dtype="bfloat16"):
    return dict(B=2, M=65, N=129, HQ=4, H=4, D=64, scale=0.125,
                dtype=dtype, reference_kind="native", id="reuse")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_forgetting_attention_launch_reuse(dtype):
    case = _reuse_case(dtype)
    inputs = make_inputs(case, 37)
    for threshold, scale in [(-10.0, 1), (-5.0, 1.0), (-10.0, 0.125), (0.0, 0.125)]:
        assert_bitwise(call_optimized(inputs, case, threshold, scale),
                       call_official(inputs, case, threshold, scale))
    fresh = tuple(t.clone() for t in inputs)
    assert_bitwise(call_optimized(fresh, case), call_official(fresh, case))
    fresh[0].mul_(0.5)
    fresh[2].neg_()
    fresh[3].mul_(1.125)
    assert_bitwise(call_optimized(fresh, case), call_official(fresh, case))


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@torch.inference_mode()
def test_forgetting_attention_gate_alignment(offset):
    case = _reuse_case()
    q, k, v, gate = make_inputs(case, 5)
    storage = torch.empty(gate.numel() + offset, device="cuda", dtype=gate.dtype)
    shifted = storage[offset:].view_as(gate)
    shifted.copy_(gate)
    inputs = (q, k, v, shifted)
    assert_bitwise(call_optimized(inputs, case), call_official(inputs, case))


@torch.inference_mode()
def test_forgetting_attention_nondefault_stream_and_lifetime():
    case = _reuse_case()
    for _ in range(2):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            inputs = make_inputs(case, 8)
            expected = call_official(inputs, case)
            out = call_optimized(inputs, case)
            assert_bitwise(out, expected)
        stream.synchronize()
        refs = [weakref.ref(t) for t in (*inputs, out)]
        del inputs, expected, out
        gc.collect()
        assert all(ref() is None for ref in refs), "launch plan retains caller tensors"


@pytest.mark.parametrize("problem", ["head_first", "seq_start", "threshold_none", "threshold_positive",
                                    "threshold_nan", "scale_inf", "gate_dtype", "q_dtype", "noncontiguous"])
@torch.inference_mode()
def test_forgetting_attention_rejects_unsupported(problem):
    case = _reuse_case()
    q, k, v, gate = make_inputs(case)
    kwargs = dict(head_first=False, seq_start=None, sm_scale=0.125, adaptive_threshold=-10.0)
    if problem == "head_first":
        kwargs["head_first"] = True
    elif problem == "seq_start":
        kwargs["seq_start"] = torch.zeros(2, device="cuda", dtype=torch.int32)
    elif problem == "threshold_none":
        kwargs["adaptive_threshold"] = None
    elif problem == "threshold_positive":
        kwargs["adaptive_threshold"] = 1.0
    elif problem == "threshold_nan":
        kwargs["adaptive_threshold"] = float("nan")
    elif problem == "scale_inf":
        kwargs["sm_scale"] = float("inf")
    elif problem == "gate_dtype":
        gate = gate.half()
    elif problem == "q_dtype":
        q = q.float()
    else:
        q = torch.empty((*q.shape[:-1], q.shape[-1] * 2), device="cuda", dtype=q.dtype)[..., ::2]
    with pytest.raises((NotImplementedError, ValueError)):
        optimized_entry()(q, k, v, gate, **kwargs)


@pytest.mark.parametrize("id", ["S031", "S034", "S024"])
@torch.inference_mode()
def test_forgetting_attention_explicit_async_and_hot_launch(id):
    from flag_attn.forgetting_attention.parallel import capture
    case = next(c for c in CASES if c["id"] == id)
    inputs = make_inputs(case)
    call_optimized(inputs, case)
    with capture() as records:
        call_optimized(inputs, case)
    torch.cuda.synchronize()
    attention = [r for r in records if "cp.async.bulk.tensor" in r[1].asm["ptx"]]
    assert len(attention) == 1
    _, compiled, _, _, cache_hit = attention[0]
    assert cache_hit
    assert ("wgmma.mma_async" in compiled.asm["ptx"]) == (case["M"] > 1)
