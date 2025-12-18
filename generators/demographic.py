from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

import sys
sys.path.insert(0, '/home/thanhdang/Desktop/vnpt-ai')

from config.settings import SyntheticDataConfig, Region
from generators.base import (
    BaseDataGenerator,
    CorrelationMixin,
    generate_vietnamese_name,
    weighted_random_choice,
)


# VIETNAMESE PROVINCE DATA

@dataclass
class ProvinceInfo:
    code: str
    name: str
    region: str
    population: int  # thousands
    urban_rate: float  # urbanization rate (0-1)


# 63 provinces/cities of Vietnam with population and urbanization data
VIETNAM_PROVINCES: List[ProvinceInfo] = [
    # Đồng bằng sông Hồng (Red River Delta)
    ProvinceInfo("HN", "Hà Nội", "dong_bang_song_hong", 8246, 0.72),
    ProvinceInfo("HP", "Hải Phòng", "dong_bang_song_hong", 2029, 0.67),
    ProvinceInfo("QN", "Quảng Ninh", "dong_bang_song_hong", 1321, 0.65),
    ProvinceInfo("BN", "Bắc Ninh", "dong_bang_song_hong", 1369, 0.58),
    ProvinceInfo("HD", "Hải Dương", "dong_bang_song_hong", 1895, 0.32),
    ProvinceInfo("HY", "Hưng Yên", "dong_bang_song_hong", 1252, 0.18),
    ProvinceInfo("TB", "Thái Bình", "dong_bang_song_hong", 1861, 0.12),
    ProvinceInfo("HNa", "Hà Nam", "dong_bang_song_hong", 853, 0.20),
    ProvinceInfo("ND", "Nam Định", "dong_bang_song_hong", 1834, 0.21),
    ProvinceInfo("NB", "Ninh Bình", "dong_bang_song_hong", 983, 0.22),
    ProvinceInfo("VP", "Vĩnh Phúc", "dong_bang_song_hong", 1152, 0.28),

    # Trung du và miền núi phía Bắc (Northern Midlands and Mountains)
    ProvinceInfo("HG", "Hà Giang", "mien_nui_phia_bac", 855, 0.14),
    ProvinceInfo("CB", "Cao Bằng", "mien_nui_phia_bac", 530, 0.23),
    ProvinceInfo("BK", "Bắc Kạn", "mien_nui_phia_bac", 314, 0.20),
    ProvinceInfo("TQ", "Tuyên Quang", "mien_nui_phia_bac", 785, 0.16),
    ProvinceInfo("LC", "Lào Cai", "mien_nui_phia_bac", 730, 0.24),
    ProvinceInfo("YB", "Yên Bái", "mien_nui_phia_bac", 821, 0.21),
    ProvinceInfo("TN", "Thái Nguyên", "mien_nui_phia_bac", 1287, 0.36),
    ProvinceInfo("LS", "Lạng Sơn", "mien_nui_phia_bac", 782, 0.22),
    ProvinceInfo("BG", "Bắc Giang", "mien_nui_phia_bac", 1804, 0.18),
    ProvinceInfo("PT", "Phú Thọ", "mien_nui_phia_bac", 1463, 0.20),
    ProvinceInfo("DB", "Điện Biên", "mien_nui_phia_bac", 598, 0.18),
    ProvinceInfo("LCa", "Lai Châu", "mien_nui_phia_bac", 461, 0.16),
    ProvinceInfo("SL", "Sơn La", "mien_nui_phia_bac", 1249, 0.15),
    ProvinceInfo("HB", "Hòa Bình", "mien_nui_phia_bac", 854, 0.17),

    # Bắc Trung Bộ (North Central Coast)
    ProvinceInfo("TH", "Thanh Hóa", "bac_trung_bo", 3641, 0.18),
    ProvinceInfo("NA", "Nghệ An", "bac_trung_bo", 3328, 0.17),
    ProvinceInfo("HT", "Hà Tĩnh", "bac_trung_bo", 1289, 0.19),
    ProvinceInfo("QB", "Quảng Bình", "bac_trung_bo", 896, 0.20),
    ProvinceInfo("QT", "Quảng Trị", "bac_trung_bo", 633, 0.30),
    ProvinceInfo("TTH", "Thừa Thiên Huế", "bac_trung_bo", 1129, 0.52),

    # Duyên hải Nam Trung Bộ (South Central Coast)
    ProvinceInfo("DN", "Đà Nẵng", "nam_trung_bo", 1134, 0.87),
    ProvinceInfo("QNa", "Quảng Nam", "nam_trung_bo", 1496, 0.26),
    ProvinceInfo("QNg", "Quảng Ngãi", "nam_trung_bo", 1232, 0.17),
    ProvinceInfo("BD", "Bình Định", "nam_trung_bo", 1487, 0.33),
    ProvinceInfo("PY", "Phú Yên", "nam_trung_bo", 875, 0.29),
    ProvinceInfo("KH", "Khánh Hòa", "nam_trung_bo", 1232, 0.52),
    ProvinceInfo("NT", "Ninh Thuận", "nam_trung_bo", 591, 0.38),
    ProvinceInfo("BTh", "Bình Thuận", "nam_trung_bo", 1231, 0.40),

    # Tây Nguyên (Central Highlands)
    ProvinceInfo("KT", "Kon Tum", "tay_nguyen", 540, 0.36),
    ProvinceInfo("GL", "Gia Lai", "tay_nguyen", 1514, 0.30),
    ProvinceInfo("DL", "Đắk Lắk", "tay_nguyen", 1870, 0.26),
    ProvinceInfo("DNg", "Đắk Nông", "tay_nguyen", 623, 0.16),
    ProvinceInfo("LDg", "Lâm Đồng", "tay_nguyen", 1297, 0.40),

    # Đông Nam Bộ (Southeast)
    ProvinceInfo("HCM", "Hồ Chí Minh", "dong_nam_bo", 9166, 0.83),
    ProvinceInfo("BDg", "Bình Dương", "dong_nam_bo", 2427, 0.82),
    ProvinceInfo("DNa", "Đồng Nai", "dong_nam_bo", 3098, 0.44),
    ProvinceInfo("BR", "Bà Rịa - Vũng Tàu", "dong_nam_bo", 1149, 0.54),
    ProvinceInfo("BP", "Bình Phước", "dong_nam_bo", 995, 0.23),
    ProvinceInfo("TNo", "Tây Ninh", "dong_nam_bo", 1170, 0.22),

    # Đồng bằng sông Cửu Long (Mekong Delta)
    ProvinceInfo("LA", "Long An", "dong_bang_scl", 1689, 0.18),
    ProvinceInfo("TG", "Tiền Giang", "dong_bang_scl", 1765, 0.16),
    ProvinceInfo("BT", "Bến Tre", "dong_bang_scl", 1289, 0.11),
    ProvinceInfo("TV", "Trà Vinh", "dong_bang_scl", 1010, 0.18),
    ProvinceInfo("VL", "Vĩnh Long", "dong_bang_scl", 1023, 0.17),
    ProvinceInfo("DT", "Đồng Tháp", "dong_bang_scl", 1600, 0.18),
    ProvinceInfo("AG", "An Giang", "dong_bang_scl", 1909, 0.31),
    ProvinceInfo("KG", "Kiên Giang", "dong_bang_scl", 1724, 0.29),
    ProvinceInfo("CT", "Cần Thơ", "dong_bang_scl", 1236, 0.70),
    ProvinceInfo("HGi", "Hậu Giang", "dong_bang_scl", 734, 0.25),
    ProvinceInfo("ST", "Sóc Trăng", "dong_bang_scl", 1200, 0.31),
    ProvinceInfo("BL", "Bạc Liêu", "dong_bang_scl", 908, 0.32),
    ProvinceInfo("CM", "Cà Mau", "dong_bang_scl", 1195, 0.24),
]


# DEMOGRAPHIC DISTRIBUTIONS

# Education level distribution by age group
EDUCATION_BY_AGE: Dict[str, Dict[str, float]] = {
    # Age 18-24: Younger generation, higher education rates
    "18-24": {
        "tieu_hoc": 0.02,
        "thcs": 0.08,
        "thpt": 0.35,
        "trung_cap": 0.12,
        "cao_dang": 0.15,
        "dai_hoc": 0.25,
        "thac_si": 0.02,
        "tien_si": 0.01,
    },
    # Age 25-34: University graduates entering workforce
    "25-34": {
        "tieu_hoc": 0.03,
        "thcs": 0.10,
        "thpt": 0.25,
        "trung_cap": 0.12,
        "cao_dang": 0.13,
        "dai_hoc": 0.30,
        "thac_si": 0.05,
        "tien_si": 0.02,
    },
    # Age 35-44: Mixed education
    "35-44": {
        "tieu_hoc": 0.05,
        "thcs": 0.15,
        "thpt": 0.28,
        "trung_cap": 0.15,
        "cao_dang": 0.12,
        "dai_hoc": 0.20,
        "thac_si": 0.04,
        "tien_si": 0.01,
    },
    # Age 45-54: Lower overall education
    "45-54": {
        "tieu_hoc": 0.08,
        "thcs": 0.20,
        "thpt": 0.30,
        "trung_cap": 0.15,
        "cao_dang": 0.10,
        "dai_hoc": 0.14,
        "thac_si": 0.02,
        "tien_si": 0.01,
    },
    # Age 55-65: Older generation
    "55-65": {
        "tieu_hoc": 0.12,
        "thcs": 0.25,
        "thpt": 0.28,
        "trung_cap": 0.13,
        "cao_dang": 0.08,
        "dai_hoc": 0.11,
        "thac_si": 0.02,
        "tien_si": 0.01,
    },
}

# Marital status distribution by age group
MARITAL_STATUS_BY_AGE: Dict[str, Dict[str, float]] = {
    "18-24": {
        "doc_than": 0.85,
        "da_ket_hon": 0.13,
        "ly_hon": 0.01,
        "goa": 0.01,
    },
    "25-34": {
        "doc_than": 0.35,
        "da_ket_hon": 0.60,
        "ly_hon": 0.04,
        "goa": 0.01,
    },
    "35-44": {
        "doc_than": 0.10,
        "da_ket_hon": 0.80,
        "ly_hon": 0.08,
        "goa": 0.02,
    },
    "45-54": {
        "doc_than": 0.05,
        "da_ket_hon": 0.82,
        "ly_hon": 0.08,
        "goa": 0.05,
    },
    "55-65": {
        "doc_than": 0.03,
        "da_ket_hon": 0.75,
        "ly_hon": 0.07,
        "goa": 0.15,
    },
}

# Education level labels in Vietnamese
EDUCATION_LABELS: Dict[str, str] = {
    "tieu_hoc": "Tiểu học",
    "thcs": "THCS",
    "thpt": "THPT",
    "trung_cap": "Trung cấp",
    "cao_dang": "Cao đẳng",
    "dai_hoc": "Đại học",
    "thac_si": "Thạc sĩ",
    "tien_si": "Tiến sĩ",
}

# Marital status labels in Vietnamese
MARITAL_STATUS_LABELS: Dict[str, str] = {
    "doc_than": "Độc thân",
    "da_ket_hon": "Đã kết hôn",
    "ly_hon": "Ly hôn",
    "goa": "Góa",
}


# DEMOGRAPHIC GENERATOR

class DemographicGenerator(BaseDataGenerator, CorrelationMixin):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(config, seed)
        self._setup_province_weights()
        self._set_default_schema()

    def _setup_province_weights(self) -> None:
        total_population = sum(p.population for p in VIETNAM_PROVINCES)
        self._province_weights = [
            p.population / total_population for p in VIETNAM_PROVINCES
        ]

    def _set_default_schema(self) -> None:
        self.set_schema({
            'customer_id': str,
            'full_name': str,
            'gender': str,
            'date_of_birth': object,  # datetime
            'age': int,
            'marital_status': str,
            'marital_status_code': str,
            'education_level': str,
            'education_level_code': str,
            'province_code': str,
            'province_name': str,
            'region': str,
            'urban_rural': str,
            'address_stability_years': float,
        })

    def _get_age_group(self, age: int) -> str:
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55-65"

    def _generate_ages(self, n: int) -> np.ndarray:
        # Vietnamese age distribution (approximated)
        # Younger population bias with working-age peak
        min_age = self.config.vietnamese_market.min_age
        max_age = self.config.vietnamese_market.max_age

        # Use beta distribution for realistic age spread
        # Parameters tuned for Vietnamese demographics (younger population)
        alpha, beta_param = 2.5, 4.0
        ages_normalized = self.rng.beta(alpha, beta_param, size=n)
        ages = min_age + ages_normalized * (max_age - min_age)

        return ages.astype(int)

    def _generate_genders(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Slight male majority (51.2% male in Vietnam)
        gender_codes = self.rng.choice(
            ['male', 'female'],
            size=n,
            p=[0.512, 0.488]
        )
        gender_labels = np.where(gender_codes == 'male', 'Nam', 'Nữ')

        return gender_codes, gender_labels

    def _generate_dates_of_birth(
        self,
        ages: np.ndarray,
        reference_date: Optional[date] = None
    ) -> np.ndarray:
        if reference_date is None:
            reference_date = date.today()

        dates = []
        for age in ages:
            # Random day within the birth year
            birth_year = reference_date.year - age
            start_date = date(birth_year, 1, 1)
            end_date = date(birth_year, 12, 31)
            days_in_year = (end_date - start_date).days
            random_day = int(self.rng.integers(0, days_in_year))
            birth_date = start_date + timedelta(days=random_day)
            dates.append(birth_date)

        return np.array(dates)

    def _generate_marital_status(
        self,
        ages: np.ndarray,
        genders: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ages)
        status_codes = np.empty(n, dtype=object)
        status_labels = np.empty(n, dtype=object)

        for i in range(n):
            age_group = self._get_age_group(ages[i])
            distribution = MARITAL_STATUS_BY_AGE[age_group].copy()

            # Slight adjustment for gender
            # Women tend to marry younger in Vietnam
            if genders[i] == 'female' and ages[i] < 30:
                distribution['da_ket_hon'] *= 1.1
                distribution['doc_than'] *= 0.9

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            status_code = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            status_codes[i] = status_code
            status_labels[i] = MARITAL_STATUS_LABELS[status_code]

        return status_codes, status_labels

    def _generate_education(
        self,
        ages: np.ndarray,
        urban_rural: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ages)
        edu_codes = np.empty(n, dtype=object)
        edu_labels = np.empty(n, dtype=object)

        for i in range(n):
            age_group = self._get_age_group(ages[i])
            distribution = EDUCATION_BY_AGE[age_group].copy()

            # Urban areas have higher education rates
            if urban_rural[i] == 'Thành thị':
                # Boost higher education
                distribution['dai_hoc'] *= 1.3
                distribution['thac_si'] *= 1.5
                distribution['tien_si'] *= 1.5
                # Reduce lower education
                distribution['tieu_hoc'] *= 0.5
                distribution['thcs'] *= 0.7

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            edu_code = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            edu_codes[i] = edu_code
            edu_labels[i] = EDUCATION_LABELS[edu_code]

        return edu_codes, edu_labels

    def _generate_provinces(
        self,
        n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Select provinces based on population weights
        province_indices = self.rng.choice(
            len(VIETNAM_PROVINCES),
            size=n,
            p=self._province_weights
        )

        province_codes = np.array([
            VIETNAM_PROVINCES[i].code for i in province_indices
        ])
        province_names = np.array([
            VIETNAM_PROVINCES[i].name for i in province_indices
        ])
        regions = np.array([
            VIETNAM_PROVINCES[i].region for i in province_indices
        ])
        urban_rates = np.array([
            VIETNAM_PROVINCES[i].urban_rate for i in province_indices
        ])

        return province_codes, province_names, regions, urban_rates

    def _generate_urban_rural(
        self,
        urban_rates: np.ndarray
    ) -> np.ndarray:
        n = len(urban_rates)
        is_urban = self.rng.random(n) < urban_rates
        return np.where(is_urban, 'Thành thị', 'Nông thôn')

    def _generate_address_stability(
        self,
        ages: np.ndarray,
        urban_rural: np.ndarray
    ) -> np.ndarray:
        n = len(ages)
        stability = np.zeros(n)

        for i in range(n):
            age = ages[i]
            is_urban = urban_rural[i] == 'Thành thị'

            # Base parameters
            if age < 25:
                mean_years = 2
                max_years = min(age - 18, 7)
            elif age < 35:
                mean_years = 4 if is_urban else 6
                max_years = 15
            elif age < 50:
                mean_years = 8 if is_urban else 12
                max_years = 25
            else:
                mean_years = 15 if is_urban else 25
                max_years = 40

            # Use gamma distribution for right-skewed stability
            shape = 2.0
            scale = mean_years / shape
            years = self.rng.gamma(shape, scale)
            years = min(years, max_years, age - 18)
            years = max(years, 0)

            stability[i] = round(years, 1)

        return stability

    def _generate_customer_ids(self, n: int) -> np.ndarray:
        return np.array([f"VN{i:08d}" for i in range(1, n + 1)])

    def _generate_full_names(
        self,
        n: int,
        gender_codes: np.ndarray
    ) -> np.ndarray:
        names = np.empty(n, dtype=object)
        for i in range(n):
            names[i] = generate_vietnamese_name(self.rng, gender=gender_codes[i])
        return names

    def generate(self) -> pd.DataFrame:
        n = self.config.credit_scoring.n_samples

        # Step 1: Generate base attributes
        customer_ids = self._generate_customer_ids(n)
        ages = self._generate_ages(n)
        gender_codes, gender_labels = self._generate_genders(n)
        dates_of_birth = self._generate_dates_of_birth(ages)

        # Step 2: Generate location (needed for education correlation)
        province_codes, province_names, regions, urban_rates = \
            self._generate_provinces(n)
        urban_rural = self._generate_urban_rural(urban_rates)

        # Step 3: Generate correlated attributes
        marital_codes, marital_labels = self._generate_marital_status(
            ages, gender_codes
        )
        edu_codes, edu_labels = self._generate_education(ages, urban_rural)

        # Step 4: Generate address stability (depends on age and location)
        address_stability = self._generate_address_stability(ages, urban_rural)

        # Step 5: Generate names based on gender
        full_names = self._generate_full_names(n, gender_codes)

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'full_name': full_names,
            'gender': gender_labels,
            'gender_code': gender_codes,
            'date_of_birth': dates_of_birth,
            'age': ages,
            'marital_status': marital_labels,
            'marital_status_code': marital_codes,
            'education_level': edu_labels,
            'education_level_code': edu_codes,
            'province_code': province_codes,
            'province_name': province_names,
            'region': regions,
            'urban_rural': urban_rural,
            'address_stability_years': address_stability,
        })

        self._generated_data = df
        return df

    def get_correlation_report(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        report = {
            "sample_size": len(data),
            "age_distribution": {},
            "marital_by_age": {},
            "education_by_urban": {},
            "province_distribution": {},
        }

        # Age distribution
        report["age_distribution"] = {
            "mean": float(data['age'].mean()),
            "std": float(data['age'].std()),
            "min": int(data['age'].min()),
            "max": int(data['age'].max()),
            "quartiles": data['age'].quantile([0.25, 0.5, 0.75]).to_dict(),
        }

        # Marital status by age group
        data['age_group'] = data['age'].apply(self._get_age_group)
        marital_crosstab = pd.crosstab(
            data['age_group'],
            data['marital_status'],
            normalize='index'
        )
        report["marital_by_age"] = marital_crosstab.to_dict()

        # Education by urban/rural
        edu_urban_crosstab = pd.crosstab(
            data['urban_rural'],
            data['education_level'],
            normalize='index'
        )
        report["education_by_urban"] = edu_urban_crosstab.to_dict()

        # Province distribution
        province_counts = data['province_name'].value_counts().head(10)
        report["province_distribution"] = province_counts.to_dict()

        # Gender distribution
        report["gender_distribution"] = data['gender'].value_counts(
            normalize=True
        ).to_dict()

        return report


# MODULE EXPORTS

__all__ = [
    "DemographicGenerator",
    "ProvinceInfo",
    "VIETNAM_PROVINCES",
    "EDUCATION_LABELS",
    "MARITAL_STATUS_LABELS",
]
