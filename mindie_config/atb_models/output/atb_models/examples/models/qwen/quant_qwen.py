# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import sys
import logging
import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained

data_list_1 = [
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法\
        治国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领\
        域制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资\
        规模较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次\
        的问题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道\
        理，是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n我国人民代表大\
        会制度组织和活动的基本原则是____\nA. 人民当家作主的原则\nB. 民主集中制的原则\nC. 在宪法和法律范围内活动的原则\nD. 公平、公正、\
        公开的原则\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n社会主义和谐社会的核心价值\
        是____\nA. 以人为本\nB. 以民为本\nC. 社会公平\nD. 公平和正义\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进行\nB\
        . 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国家进\
        行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围绕____\nA. \
        推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制度创新\nAnsw\
        er: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较小\nB. 经营方式\
        单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。公有制经济的性质\
        体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执政兴国的第一要\
        务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n21世纪前10年，我国经济体制必须解决好的\
        历史课题是____\nA. 实施科教兴国战略和可持续发展战略\nB. 促进国民经济持续快速健康发展\nC. 大多数国有大中型骨干企业建立现代企业制\
        度\nD. 建立比较完善的社会主义市场经济体制\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n首次以“台湾回到祖国怀抱，\
        实现统一大业”来代替“解放台湾”的提法的是____\nA. 1978年12月党的十一届三中全会公报\nB. 1979年元旦全国人大常委会发表《告台湾同胞书\
        》\nC. 1981年9月叶剑英对新华社记者发表的被称为“叶九条”的谈话\nD. 1982年中国共产党十二大政治报告\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n毛泽东思想达到成熟的标志\
        是____\nA. 新民主主义理论科学体系的形成\nB. 农村包围城市革命道路理论的形成\nC. 新民主主义革命基本经验的提出\nD. 毛泽东军事路线的\
        完整形成\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进行\nB\
        . 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国家进\
        行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围绕____\nA. \
        推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制度创\
        新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题\
        。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党\
        执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n建设和谐文化的根本\
        是____\nA. 坚持马克思主义的指导\nB. 发展科学和教育\nC. 坚持社会主义核心价值体系\nD. 推进文化体制改革\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党\
        执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n构建社会主义和谐社会的重\
        点是____\nA. 坚持以马列主义、毛泽东思想、邓小平理论和“三个代表”重要思想为指导\nB. 民主法制、公平正义、诚信友爱、充满活力、安定有\
        序、人与自然和谐相处\nC. 解决人民群众最关心、最直接、最现实的利益问题\nD. 到2020年完全实现社会主义和谐社会\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n台湾问题的本质是____\nA. \
        中国的内政问题\nB. 中国同美国的关系问题\nC. 中国同日本的关系问题\nD. 共产党与国民党的关系问题\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n在中国共产党历史上，最早提\
        出“马克思主义中国化”这个命题的是____\nA. 李大钊\nB. 陈独\nC. 毛泽东\nD. 张闻天\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n党的十八大提出，面对资源约\
        束趋紧、环境污染严重、生态系统退化的严峻形势，必须树立尊重自然、顺应自然、保护自然的生态文明理念。人与自然相处时应秉持的首要态度\
        是____\nA. 尊重自然\nB. 顺应自然\nC. 保护自然\nD. 征服自然\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域制\
        度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党\
        执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n中国革命的特点和优点\
        是____\nA. 由中国共产党领导的人民战争\nB. 目标是争取民族独立、人民解放，最终实现国家的繁荣富强\nC. 以反帝反封建作为两大革命任\
        务\nD. 以武装的革命反对武装的反革命\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过的\
        《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法\
        治国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规\
        模较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的\
        问题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道\
        理，是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n香港特别行政区\
        的高度自治权的唯一来源是____\nA. 中央授权\nB. 香港特别行政区本身固有的\nC. 《中英联合声明》\nD. 中央授权之外的剩余权\
        力\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模\
        较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问\
        题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，\
        是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n我国实施改革的目\
        的是____\nA. 巩固社会主义制度\nB. 发扬社会主义民主\nC. 调动广大人民群众的积极性\nD. 发展社会主义的生产力\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模\
        较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问\
        题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，\
        是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n新民主主义革命总路\
        线的核心是____\nA. 无产阶级的领导\nB. 人民大众的参与\nC. 反帝反封建\nD. 反官僚资本主义\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模\
        较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问\
        题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，\
        是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n九届人大二次会议正式\
        将“依法治国”写入宪法，这一政策的核心是____\nA. 人民当家作主\nB. 民主与法制的结合\nC. 法治代替人治\nD. 有法可依，有法必依，执法\
        必严，违法必究\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模\
        较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问\
        题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，\
        是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n在社会主义初级阶段，\
        非公有制经济是____\nA. 社会主义公有制经济的补充\nB. 社会主义市场经济的重要组成部分\nC. 具有公有性质的经济\nD. 逐步向公有制过渡的\
        经济\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规\
        模较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的\
        问题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理\
        ，是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n过渡时期总路线的\
        特征是____\nA. 重视工业建设\nB. 强调三大改造\nC. 社会主义建设和社会主义改造同时并举\nD. 尤其重视对资本主义工商业的改\
        造\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进行\
        \nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治国\
        家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题\
        。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是\
        党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n毛泽东思想开始形成是\
        在____\nA. 国民革命时期\nB. 土地革命战争时期\nC. 解放战争时期\nD. 抗日战争时期\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通过\
        的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模\
        较小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问\
        题。公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，\
        是党执政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n当今时代的主题\
        是____\nA. 战争与革命\nB. 和平与发展\nC. 开放与合作\nD. 和谐与共赢\nAnswer:",
        "The following are multiple choice questions (with answers) about  mao zedong thought.\n\n中共十八届三中全会通\
        过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化政治体制改革要紧紧围绕____\nA. 提高科学执政、民主执政、依法执政水平进\
        行\nB. 坚持党的领导、人民当家作主、依法治国有机统一进行\nC. 推进社会主义民主政治制度化、规范化、程序化进行\nD. 建设社会主义法治\
        国家进行\nAnswer: B\n\n中共十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》指出，深化社会体制改革要紧紧围\
        绕____\nA. 推进基本公共服务均等化\nB. 改革收入分配制度，促进共同富裕\nC. 更好保障和改善民生、促进社会公平正义\nD. 推进社会领域\
        制度创新\nAnswer: C\n\n个体经济、私营经济都是非公有制经济，但是，个体经济在性质上不同于私营经济，因为个体经济____\nA. 投资规模较\
        小\nB. 经营方式单一\nC. 主要依靠自己劳动和经营\nD. 不是法人企业\nAnswer: C\n\n公有制经济的性质和实现形式是两个不同层次的问题。\
        公有制经济的性质体现在____\nA. 组织形式上\nB. 所有权的归属上\nC. 经营方式上\nD. 分配方式上\nAnswer: B\n\n发展是硬道理，是党执\
        政兴国的第一要务。要把发展的基点放在____\nA. 改革上\nB. 科技上\nC. 创新上\nD. 制度上\nAnswer: C\n\n正式把毛泽东思想确立为党的\
        指导思想并首次写进党章的是____\nA. 中共六大\nB. 中共七大\nC. 中共八大\nD. 中共十二大\nAnswer:"
    ]

data_list_2 = [
        ["电子发票有哪些注意事项？"],
        ["费用报销需要提供哪些材料？"],
        ["微信支付可以支持哪些银行卡？"],
        ["简历中应该如何突出重点？"],
        ["海外留学需要注意哪些事项？"],
        ["云计算对于企业有哪些好处？"],
        ["常见的投资方式有哪些？"],
        ["什么是股票的基本面分析？"],
        ["运动员如何保持良好的竞技状态？"],
        ["暴雨天气应该注意哪些安全事项？"],
        ["驾照考试一共有几个科目？"],
        ["食品安全检测的流程是什么？"],
        ["股票交易中的龙头股是什么？"],
        ["网络攻击有哪些形式？"],
        ["新能源汽车的优势是什么？"],
        ["What are the benefits of cloud computing for businesses?"],
        ["What documents are required for expense reimbursement?"],
        ["How to highlight key points in a resume?"],
        ["What should be paid attention to when studying abroad?"],
        ["Which banks does WeChat payment support?"],
        ["What are the common investment methods?"],
        ["What is the process of food safety inspection?"],
        ["What is the basic analysis of stock fundamentals?"],
        ["How do athletes maintain good athletic performance?"],
        ["What safety precautions should be taken in rainy weather?"],
        ["What are the subjects of the driver's license exam?"],
        ["What are the types of cyber attacks?"],
        ["What is the concept of leading stocks in stock trading?"],
        ["What should be noted in the use of electronic invoices?"],
        ["What are the advantages of new energy vehicles?"],
        ["如何有效管理个人财务？"],
        ["什么是人工智能的发展趋势？"],
        ["如何设计一个用户友好的网站界面？"],
        ["为什么要进行环境保护？"],
        ["如何预防常见的网络安全漏洞？"],
        ["如何培养良好的沟通能力？"],
        ["学习一门外语需要多长时间？"],
        ["什么是健康的饮食习惯？"],
        ["什么是心理健康？如何保持心理健康？"],
        ["如何应对工作压力？"],
        ["How to effectively manage personal finances?"],
        ["What are the development trends of artificial intelligence?"],
        ["How to design a user-friendly website interface?"],
        ["Why is environmental protection important?"],
        ["How to prevent common network security vulnerabilities?"],
        ["How to cultivate good communication skills?"],
        ["How long does it take to learn a foreign language?"],
        ["What are healthy eating habits?"],
        ["What is mental health and how to maintain it?"],
        ["How to cope with work-related stress?"]
    ]


def load_tokenizer_and_model(model_path, trust_remote_code=False):
    tokenizer = safe_get_tokenizer_from_pretrained(
        model_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=trust_remote_code
    )
    model = safe_get_model_from_pretrained(
        model_path,
        torch_dtype=torch.float32, trust_remote_code=trust_remote_code
    ).cpu()
    return tokenizer, model


def infer(tokenizer, model, query, model_params=None):
    """
    推理代码
    :param query:
    :param model_params:
    :return:
    """
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    with torch.no_grad():
        model_params = model_params if model_params is not None else {}
        pred = model.generate(**inputs, **model_params)
    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return output


def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to("cpu")
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


def check_device_type(value):
    valid_device_types = {"npu", "cpu"}
    if value in valid_device_types:
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} is not a valid device type. Choose from {valid_device_types}")


def print_args(args):
    for arg in vars(args):
        value = getattr(args, arg)
        value_type = type(value)
        logging.info(f"Argument: {arg}, Value: {value}, Type: {value_type}")


def parse_arguments():
    store_true = 'store_true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str, help="quant weight path")
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--act_method", type=int, default=3)
    parser.add_argument('--disable_level', type=str, help="disable_level L0~L5", default='L0')
    parser.add_argument('--device_type', type=check_device_type, required=True, help="device type npu/cpu")
    parser.add_argument("--data_list_index", type=int, default=0)
    parser.add_argument(
        '--disable_names',
        nargs='+',
        default=["lm_head"], required=True)
    parser.add_argument('--anti_method', type=str, default=None)
    parser.add_argument('--w_sym', action=store_true)
    args = parser.parse_args()
    # 查看输出 print_args
    return args

selected_list = {
    1 : data_list_1,
    2 : data_list_2
}

if __name__ == '__main__':
    args = parse_arguments()
    input_dict = {
        "model_path":args.model_path,
        "trust_remote_code" : False
    }
    tokenizer, model = load_tokenizer_and_model(**input_dict)
    data_list = selected_list.get(args.data_list_index)
    if data_list is None:
        raise ValueError(f"Selected list is None, invalid key. {args.data_list_index}")
    dataset_calib = get_calib_dataset(tokenizer, data_list)
    disable_names = args.disable_names
    if args.anti_method:
        # dev_type="npu", dev_id=0  如果需要使用npu进行量化
        anti_config = AntiOutlierConfig(
            anti_method=args.anti_method,
            dev_type=args.device_type
        )
        anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
        anti_outlier.process()
    quant_config = QuantConfig(
        w_bit=args.w_bit,  # 权重量化位数
        a_bit=args.a_bit,  # 激活值量化位数
        disable_names=disable_names,  # 不做量化的层（通常是空list）
        dev_type=args.device_type,
        act_method=args.act_method,  # 激活量化方法，建议方法3（1：min-max；2：histogram；3：自动混合量化）
        pr=1.0,  # 量化正则百分比，建议0.5
        w_sym=args.w_sym,  # 对称/非对称量化，True为对称量化，False为非对称量化
        mm_tensor=False  # 权重量化粒度，True为per-tensor量化，False为per-channel量化（大模型场景建议False）
    )
    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=dataset_calib,
        disable_level=args.disable_level  # 自动回退等级，根据精度损失程度增加不量化的层（L0~L5，L0为不回退，精度损失明显时可适当提升等级）
    )

    calibrator.run()  # 执行PTQ量化校准

    calibrator.save(args.save_directory, save_type=["safe_tensor"])   





