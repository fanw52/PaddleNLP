# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import jsonlines

__all__ = ['BussinessLicenseNer']


class BussinessLicenseNer(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
                            names=['O','B-住所', 'B-名称', 'B-法定代表人', 'B-注册号', 'B-注册日期', 'B-注册资本', 'B-类型', 'B-组成形式', 'B-经营场所',
                                   'B-经营者', 'B-经营范围', 'B-统一社会信用代码', 'B-营业期限', 'I-住所', 'I-名称', 'I-法定代表人', 'I-注册号',
                                   'I-注册日期', 'I-注册资本', 'I-类型', 'I-组成形式', 'I-经营场所', 'I-经营者', 'I-经营范围', 'I-统一社会信用代码',
                                   'I-营业期限']

                        )
                    )

                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={
                                        "filepaths": "/data/wufan/data/NERData/business_license/paddlenlp_data/train.jsonl"
                                    },
                                    ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": "/data/wufan/data/NERData/business_license/paddlenlp_data/valid.jsonl"
                },
            )
        ]

    def _generate_examples(self, filepaths):
        with jsonlines.open(filepaths) as reader:
            id = 0
            for line in reader:
                tokens = line["tokens"]
                ner_tags = line["ner_tags"]
                yield id,{"tokens": tokens, "ner_tags": ner_tags}
                id+=1

