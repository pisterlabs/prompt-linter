import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage


class SqlTunner:
    def __init__(self):
        self.init_dotenv()
        self.init_output_parser()

    def init_dotenv(self):
        _ = load_dotenv(find_dotenv())

    def init_output_parser(self):
        response_schemas = [
            ResponseSchema(name="tuned_sql",
                           description="Optimized SQL query"),
            ResponseSchema(name="what_changed",
                           description="Explanation of Changes and Reasoning"),
            ResponseSchema(name="index_suggestion",
                           description="Additional Index Suggestions"),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas)

    def get_prompt(self):
        return """
            Your task is optimizing a given SQL query to achieve better performance on TiDB while ensuring it retains the same semantics. You will receive the SQL query with the related table schema and execution plan, all enclosed in triple backquotes (`).
            If table schemas or execution plans are not provided, you may assume that they are not available, and you should try your best to optimize the SQL query without them.
            Your goal is to follow the step-by-step instructions below to improve the SQL query's execution speed.

            Step 1: Familiarize Yourself with TiDB Tuning
            To start, read the documentation of TiDB to understand how to optimize SQL performance on this platform effectively.

            Step 2: Study Table Schemas and Indexes
            Carefully examine the table schemas to understand the structure of the tables. Pay particular attention to the design of indexes, as they play a vital role in optimizing query execution.

            Step 3: Analyze the SQL Query
            Read and understand the SQL query provided. This will give you insight into the purpose and functionality of the query.

            Step 4: Understand the Execution Plan
            Analyze the execution plan of the SQL query to identify potential bottlenecks and areas for improvement.

            Step 5: Generate the Tuned SQL Query
            Based on the insights gained from the previous steps, rewrite and optimize the SQL query to enhance its performance on TiDB. You may use hints to guide the optimizer to generate a better execution plan, but using temporary tables is not allowed.

            Step 6: Explain Changes and Reasoning
            Document the changes made to the SQL query and provide a clear explanation of why each change was made. Justify how these alterations contribute to improved performance.

            Step 7: Additional Index Suggestions
            Recommend additional index designs that could further enhance the performance of the SQL query. Explain why these index suggestions are beneficial.

            After you complete all these tasks, Output the optimized SQL query, explanation of changes and reasoning, and addtional index suggestions using the following format:
            {format_instructions}

            Table Schemas: ```{schemas}```
            SQL Query: ```{sql}```
            Execution Plan: ```{execution_plan}```

        """

    def tune(self, gpt_version, prompt, original_sql, schemas, original_plan):
        prompt = prompt.format(sql=original_sql, schemas=schemas, execution_plan=original_plan,
                               format_instructions=self.output_parser.get_format_instructions())
        output = None
        try:
            chat = self.get_chat(gpt_version)
            output = chat([HumanMessage(content=prompt)])
            output = output.content.replace('\n', ' ').replace('\t', ' ')
            index = output.index('```json')
            output = output[index:]
            return self.output_parser.parse(output), prompt, output
        except Exception as e:
            print(e)
            return {"tuned_sql": "", "what_changed": "something error happended: " + str(e), "index_suggestion": ""}, prompt, output

    def get_chat(self, gpt_version):
        return ChatOpenAI(temperature=0.0, model=gpt_version, verbose=True)


if __name__ == "__main__":
    original_sql = """
        SELECT
        count(0)
        FROM
        (
            SELECT
            CUST_NO custNo,
            CUS_NAME cusName,
            acc.ACCT_NO acctNo,
            BLG_ORG_REFNO blgOrgRefno,
            BLG_ORG_NAME blgOrgName,
            OPEN_ACCT_ORG_NO openAcctOrgNo,
            ORG_NAME orgName,
            ACCT_STATUS_CD acctStatusCd,
            MCA_IND mcaInd,
            ACCT_IDNT acctIdnt,
            ACCT_FLAG acctFlag,
            OPEN_ACCT_DT openAcctDt,
            CLOSE_ACCT_DT closeAcctDt,
            PROD_TYPE_CD prodTypeCd,
            PROD_SUB_TYPE_CD prodSubTypeCd,
            PROD_SUB_TYPE_DESC prodSubTypeDesc,
            CONVERT(ACCT_BAL, CHAR) acctBal,
            CURR_CD currCd,
            OWS_CODE owsCode,
            ACC_SSPCS_FLAG accSspcsFlag,
            PUBSC_ITCO_ACC_IDR pubccItcoAccIdr,
            STPAY_IDR stpayIdr,
            ACC_SOURCE_TYPE accSourceType,
            MAIN_ACCT_FLAG mainAcctFlag
            FROM
            DWD_COM_ACC_QUERY acc
            LEFT JOIN AML_PARAMETER_RED_NAMELIST redNamelist ON CUST_NO = redNamelist.AML_RED_NAMELIST_CUSNO
            WHERE
            (
                redNamelist.AML_RED_NAMELIST_USNO = "8069443"
                OR redNamelist.AML_RED_NAMELIST_CUSNO IS NULL
            )
            AND (
                acc.ACCT_NO = "182755019311"
                OR acc.ACCT_NO IN (
                SELECT
                    card.ACCT_NO
                FROM
                    DWD_COM_CARD card
                WHERE
                    card.CARD_NO = "182755019311"
                )
            )
        ) table_count
    """
    schemas = """
        create database if not exists `sz_online_com`; use `sz_online_com`;CREATE TABLE `DWD_COM_CARD` (
        `ACCT_NO` varchar(100) NOT NULL COMMENT '账户',
        `CARD_NO` varchar(100) NOT NULL COMMENT '卡号',
        `CARD_TYPE` varchar(10) DEFAULT NULL COMMENT '卡类型',
        `OPEN_CARD_DT` varchar(10) DEFAULT NULL COMMENT '开卡日期',
        `CNCCR_DT` varchar(10) DEFAULT NULL COMMENT '销卡日期',
        `CARD_ISSUE_SER_NO` varchar(40) DEFAULT NULL COMMENT '借记卡序号',
        `CARD_MATU_DT` varchar(10) DEFAULT NULL COMMENT '借记卡有效期',
        `CARD_LIFE_STATUS_CD` varchar(60) DEFAULT NULL COMMENT '借记卡状态',
        `AML_DATA_DATE` varchar(10) DEFAULT NULL COMMENT '批量日期',
        PRIMARY KEY (`ACCT_NO`,`CARD_NO`) /*T![clustered_index] NONCLUSTERED */,
        KEY `com_idx_card_no` (`CARD_NO`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='卡信息'
        create database if not exists `sz_online_com`; use `sz_online_com`;CREATE TABLE `AML_PARAMETER_RED_NAMELIST` (
        `AML_RED_NAMELIST_CUSNO` varchar(10) DEFAULT NULL COMMENT '反洗钱红名单客户号',
        `AML_RED_NAMELIST_USNO` varchar(7) DEFAULT NULL COMMENT '反洗钱红名单用户号'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='红名单参数表'
        create database if not exists `sz_online_com`; use `sz_online_com`;CREATE TABLE `DWD_COM_ACC_QUERY` (
        `ACCT_NO` varchar(100) NOT NULL COMMENT '账户',
        `PROD_TYPE_CD` varchar(60) DEFAULT NULL COMMENT '产品类型',
        `OPEN_ACCT_ORG_NO` varchar(500) DEFAULT NULL COMMENT '账户行号',
        `ORG_NAME` varchar(500) DEFAULT NULL COMMENT '账户行机构名',
        `CUST_NO` varchar(100) DEFAULT NULL COMMENT '客户号',
        `CUS_NAME` varchar(200) DEFAULT NULL COMMENT '客户姓名',
        `PROD_SUB_TYPE_CD` varchar(60) DEFAULT NULL COMMENT '产品子类型',
        `PROD_SUB_TYPE_DESC` varchar(500) DEFAULT NULL COMMENT '产品子类型描述',
        `ACCT_STATUS_CD` varchar(60) DEFAULT NULL COMMENT '账户状态',
        `OWS_CODE` varchar(60) DEFAULT NULL COMMENT '联名标识',
        `ACCT_BAL` decimal(28,8) DEFAULT NULL COMMENT '账户余额',
        `CURR_CD` varchar(60) DEFAULT NULL COMMENT '账户币种',
        `ACCT_IDNT` varchar(60) DEFAULT NULL COMMENT '账户标识',
        `ACCT_FLAG` varchar(60) DEFAULT NULL COMMENT '账户性质',
        `CNRG_CODE` varchar(60) DEFAULT NULL COMMENT '国家/地区',
        `SUB_ACCT_CNT` varchar(10) DEFAULT NULL COMMENT '子账户数量 ',
        `CARD_CNT` varchar(60) DEFAULT NULL COMMENT '卡数量',
        `MCA_IND` varchar(60) DEFAULT NULL COMMENT '账户类型',
        `MAIN_ACCT_FLAG` varchar(60) DEFAULT NULL COMMENT '主账户标识',
        `ACCT_PARM_FCY_CASH_TRF` varchar(60) DEFAULT NULL COMMENT '钞汇标识',
        `DIAMOND` varchar(8) DEFAULT NULL COMMENT '账户分类标识',
        `ACCT_PARM_FOREIGN_PROPERTY` varchar(60) DEFAULT NULL COMMENT '外汇账户性质',
        `OPNAC_TLR_REFNO` varchar(7) DEFAULT NULL COMMENT '开户柜员编号',
        `OPEN_ACCT_DT` varchar(10) DEFAULT NULL COMMENT '开户日期',
        `PROV_BANK_ORG_NO` varchar(500) DEFAULT NULL COMMENT '省行机构号',
        `MNUL_SET_STS_CD` varchar(1) DEFAULT NULL COMMENT '信用卡账户状态',
        `PROD_DESCR` varchar(500) DEFAULT NULL COMMENT '产品描述',
        `ACCT_BAL_USD` decimal(28,8) DEFAULT NULL COMMENT '账户余额（折美元）',
        `ACCT_BAL_CNY` decimal(28,8) DEFAULT NULL COMMENT '账户余额（折人民币）',
        `DEP_ACC_ENDOF_BAL` decimal(19,3) DEFAULT NULL COMMENT '月末余额',
        `LGTRM_NO_TXN_IDR` varchar(1) DEFAULT NULL COMMENT '长期无交易标识',
        `MTAP_IDR` varchar(1) DEFAULT NULL COMMENT '一号多人标识',
        `FRZ_AMT` decimal(28,8) DEFAULT NULL COMMENT '冻结金额',
        `ACC_SSPCS_FLAG` varchar(60) DEFAULT NULL COMMENT '可疑账户标志',
        `PUBSC_ITCO_ACC_IDR` varchar(1) DEFAULT NULL COMMENT '公安涉案账户标识',
        `VRFY_STS` varchar(1) DEFAULT NULL COMMENT '公安涉案关联账户核实状态',
        `CLOSE_ACCT_DT` varchar(10) DEFAULT NULL COMMENT '关户日期',
        `STPAY_IDR` varchar(10) DEFAULT NULL COMMENT '止付标志',
        `LGTRM_NO_TXN_XIE_DT` varchar(10) DEFAULT NULL COMMENT '长期无交易解标日期',
        `OPEN_LOAN_ACCT_DT` varchar(10) DEFAULT NULL COMMENT '申请日期',
        `APPRV_DT` varchar(10) DEFAULT NULL COMMENT '审批日期',
        `REPAY_FREQ` varchar(60) DEFAULT NULL COMMENT '还款频率',
        `NEXT_RPINT_DT` varchar(10) DEFAULT NULL COMMENT '合同约定下一还款日',
        `MAIN_ACCT_NO` varchar(100) DEFAULT NULL COMMENT '对应的主账号',
        `LAST_MOD_DT` varchar(10) DEFAULT NULL COMMENT '最后修改INVM的日期',
        `LAST_FIN_TXN_DT` varchar(10) DEFAULT NULL COMMENT '上次金融交易日期',
        `CRDACC_CONN_CODE` varchar(60) DEFAULT NULL COMMENT '卡户勾连标示',
        `SML_BAL_FEE_AP_FLG` varchar(60) DEFAULT NULL COMMENT '小额账户收费',
        `SMS_PLTFM_SIGN_FLAG` varchar(10) DEFAULT NULL COMMENT '短信平台签约标识',
        `INT_DRAW_TYPE_CD` varchar(60) DEFAULT NULL COMMENT '利息支取方式',
        `INTTR_ACCNO` varchar(100) DEFAULT NULL COMMENT '转息账户',
        `RGL_DPSIT_PERIOD_START_DT` varchar(10) DEFAULT NULL COMMENT '定期存期开始日',
        `RGL_DPSIT_PERIOD_MATU_DT` varchar(10) DEFAULT NULL COMMENT '定期存期到期日',
        `RGL_DPSIT_TERM` varchar(100) DEFAULT NULL COMMENT '定期存期',
        `AUTO_TFRDE_FLAG` varchar(10) DEFAULT NULL COMMENT '是否自动转存标示',
        `PSBK_STS` varchar(10) DEFAULT NULL COMMENT '存折状态',
        `VOUCHER_TYPE` varchar(10) DEFAULT NULL COMMENT '凭证类型',
        `VCHR_REFNO` varchar(100) DEFAULT NULL COMMENT '凭证号码',
        `PSBK_VLMNO` varchar(10) DEFAULT NULL COMMENT '存折册号(定一本)',
        `RCRD_SN` varchar(10) DEFAULT NULL COMMENT '册内序号(定一本)',
        `OPNAC_DT` varchar(10) DEFAULT NULL COMMENT '客户创建日期',
        `BLG_ORG_REFNO` varchar(5) DEFAULT NULL COMMENT '客户所属机构',
        `BLG_ORG_NAME` varchar(500) DEFAULT NULL COMMENT '客户所属机构名称',
        `BLG_ORG_LVL2_REFNO` varchar(5) DEFAULT NULL COMMENT '客户所属二级行',
        `BLG_ORG_LVL2_NAME` varchar(500) DEFAULT NULL COMMENT '客户所属二级行名称',
        `DOCTYP_CD` varchar(60) DEFAULT NULL COMMENT '证件类型',
        `DOC_NO` varchar(100) DEFAULT NULL COMMENT '证件号',
        `GND_CODE` varchar(1) DEFAULT NULL COMMENT '性别',
        `BRTH_DT` varchar(10) DEFAULT NULL COMMENT '出生年月日',
        `ADDR_1` varchar(400) DEFAULT NULL COMMENT '家庭地址',
        `FAMILY_PHONE_NO` varchar(500) DEFAULT NULL COMMENT '家庭电话',
        `WORK_PHONE_NO` varchar(500) DEFAULT NULL COMMENT '办公电话',
        `MOBILE_NO` varchar(500) DEFAULT NULL COMMENT '移动电话',
        `ACC_PY_MOD_CODE` varchar(10) DEFAULT NULL COMMENT '凭证支付方式',
        `APPG_DATE` varchar(10) DEFAULT NULL COMMENT '更新日期',
        `BOC_EMPLOYEE_FLAG` varchar(1) DEFAULT NULL COMMENT '员工系统标识',
        `LAST_LOAN_FIN_TXN_DT` varchar(10) DEFAULT NULL COMMENT '上次贷方金融交易日',
        `LAST_DEBIT_FIN_TXN_DT` varchar(10) DEFAULT NULL COMMENT '上次借方金融交易日',
        `LST_MAINT_DT` varchar(10) DEFAULT NULL COMMENT '上次维护日期',
        `EBANK_REG_ACCNT_FLAG` varchar(10) DEFAULT NULL COMMENT '网银注册账户标志',
        `OPEN_EBANK_ACCT_ORG_NO` varchar(500) DEFAULT NULL COMMENT '网银账户开户行机构号',
        `MATN_EBANK_ORG_NO` varchar(500) DEFAULT NULL COMMENT '网银账户维护行机构号',
        `STMT_CYCLE` varchar(10) DEFAULT NULL COMMENT '对帐单频率',
        `STMT_FREQ` varchar(10) DEFAULT NULL COMMENT '对帐单周期',
        `STMT_DAY_CODE` varchar(10) DEFAULT NULL COMMENT '对帐单日',
        `INTACR_START_DT` varchar(10) DEFAULT NULL COMMENT '起息日',
        `INTACR_MATU_DT` varchar(10) DEFAULT NULL COMMENT '结息日',
        `CUST_INT_RATE` decimal(28,12) DEFAULT NULL COMMENT '议价利率',
        `DPSIT_CLS_OD_ACCT_FLAG` varchar(10) DEFAULT NULL COMMENT '存款类透支帐户标志',
        `WORK_ADDR_1` varchar(400) DEFAULT NULL COMMENT '办公地址',
        `CONTACT_NAME` varchar(400) DEFAULT NULL COMMENT '联系人',
        `BSN_SPEC_CODE` varchar(60) DEFAULT NULL COMMENT '业务种类',
        `DCL_FLAG` varchar(10) DEFAULT NULL COMMENT '申报标识',
        `FX_ACC_ATR_CODE` varchar(60) DEFAULT NULL COMMENT '外汇账户属性',
        `ACC_EFF_DT` varchar(10) DEFAULT NULL COMMENT '帐户生效日',
        `AGRMT_CTR_AMT` decimal(28,8) DEFAULT NULL COMMENT '协定合同金额',
        `FRNCY_ACC_PRJ_CODE` varchar(60) DEFAULT NULL COMMENT '外币帐户项目',
        `CTOF_DT` varchar(10) DEFAULT NULL COMMENT '取现有效期',
        `ENCSH_QUOTA` decimal(19,3) DEFAULT NULL COMMENT '取现额度',
        `DBT_INT_ADJ_AMT` decimal(28,8) DEFAULT NULL COMMENT '借方利息调整',
        `CR_INT_ADJ_AMT` decimal(28,8) DEFAULT NULL COMMENT '贷方利息调整',
        `TERM_BASIS` varchar(60) DEFAULT NULL COMMENT '定期支付频率是按日计还是按月计',
        `TRM_INT_ACCR` decimal(28,8) DEFAULT NULL COMMENT '存本取息应付而未付利息',
        `ACC_SOURCE_TYPE` varchar(60) NOT NULL COMMENT '账户来源',
        `LP_ORG_NO` varchar(100) DEFAULT NULL COMMENT '银行号',
        `LN_SHTNM` varchar(3) DEFAULT NULL COMMENT '贷款简称',
        `APLY_AMT` decimal(28,8) DEFAULT NULL COMMENT '申请金额',
        `DOLY_LN_BAL` decimal(19,3) DEFAULT NULL COMMENT '上年末贷款余额',
        `APPRV_AMT` decimal(28,8) DEFAULT NULL COMMENT '核准金额',
        `LOAN_DT` varchar(30) DEFAULT NULL COMMENT '首次放款日',
        `RCTLY_LN_DSBDT` varchar(30) DEFAULT NULL COMMENT '最近贷款发放日',
        `LOAN_TERM` varchar(60) DEFAULT NULL COMMENT '贷款期限，单位为月',
        `LN_EXP_DT` varchar(30) DEFAULT NULL COMMENT '贷款到期日',
        `CTR_TP_CODE` varchar(60) DEFAULT NULL COMMENT '合同类型',
        `MAIN_SUB_ACCNT_IDNT_CD` varchar(60) DEFAULT NULL COMMENT '贷款子账户标识',
        `ACCNT_TYPE_CD` varchar(60) DEFAULT NULL COMMENT '账户类型代码',
        `AML_DATA_DATE` varchar(10) DEFAULT NULL COMMENT '批量日期',
        `ACC_AML_TYPE` varchar(60) DEFAULT NULL COMMENT '账户类型代码',
        PRIMARY KEY (`ACCT_NO`,`ACC_SOURCE_TYPE`) /*T![clustered_index] NONCLUSTERED */,
        KEY `com_idx_cust_no` (`CUST_NO`),
        KEY `com_idx_acct_no_cust_no` (`ACCT_NO`,`CUST_NO`),
        KEY `com_idx_open_acct_org_no` (`OPEN_ACCT_ORG_NO`),
        KEY `com_idx_acct_status_cd` (`ACCT_STATUS_CD`),
        KEY `com_idx_mca_ind` (`MCA_IND`),
        KEY `com_idx_acct_idnt` (`ACCT_IDNT`),
        KEY `com_idx_open_acct_dt` (`OPEN_ACCT_DT`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='账户综合信息'
    """
    original_plan = """
        +----------------------------------------+---------------+------------+-----------+------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+---------+
        | id                                     | estRows       | actRows    | task      | access object                                              | execution info                                                                                                                                                                                                                                                                                                                                                                               | operator info                                                                                                                                              | memory    | disk    |
        +----------------------------------------+---------------+------------+-----------+------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+---------+
        | HashAgg_15                             | 1.00          | 1          | root      |                                                            | time:1m46.2s, loops:2, partial_worker:{wall_time:1m46.246212288s, concurrency:5, task_num:0, tot_wait:8m51.230949332s, tot_exec:0s, tot_time:8m51.230953041s, max:1m46.246192882s, p95:1m46.246192882s}, final_worker:{wall_time:1m46.246218711s, concurrency:5, task_num:0, tot_wait:8m51.23101556s, tot_exec:5.876µs, tot_time:8m51.231024726s, max:1m46.246205895s, p95:1m46.246205895s}  | funcs:count(0)->Column#135                                                                                                                                 | 148.4 KB  | N/A     |
        | └─Selection_17                         | 656408576.00  | 0          | root      |                                                            | time:1m46.2s, loops:1                                                                                                                                                                                                                                                                                                                                                                        | or(eq(sz_online_com.dwd_com_acc_query.acct_no, "182755019311"), Column#133)                                                                                | 33.7 KB   | N/A     |
        |   └─HashJoin_29                        | 820510720.00  | 1025638257 | root      |                                                            | time:34.8s, loops:1001601, build_hash_table:{total:1.08ms, fetch:1.08ms, build:0s}, probe:{concurrency:5, total:8m51.2s, max:1m46.2s, probe:4m21.6s, fetch:4m29.7s}                                                                                                                                                                                                                          | left outer semi join, equal:[eq(sz_online_com.dwd_com_acc_query.acct_no, sz_online_com.dwd_com_card.acct_no)]                                              | 0 Bytes   | 0 Bytes |
        |     ├─IndexLookUp_49(Build)            | 1.00          | 0          | root      |                                                            | time:1.04ms, loops:1                                                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                            | 217 Bytes | N/A     |
        |     │ ├─IndexRangeScan_47(Build)       | 1.00          | 0          | cop[tikv] | table:card, index:com_idx_card_no(CARD_NO)                 | time:741µs, loops:1, cop_task: {num: 1, max: 708.4µs, proc_keys: 0, rpc_num: 1, rpc_time: 694.5µs, copr_cache_hit_ratio: 0.00, distsql_concurrency: 15}, tikv_task:{time:0s, loops:1}, scan_detail: {total_keys: 1, get_snapshot_time: 21.2µs, rocksdb: {block: {cache_hit_count: 12}}}                                                                                                      | range:["182755019311","182755019311"], keep order:false                                                                                                    | N/A       | N/A     |
        |     │ └─TableRowIDScan_48(Probe)       | 1.00          | 0          | cop[tikv] | table:card                                                 |                                                                                                                                                                                                                                                                                                                                                                                              | keep order:false                                                                                                                                           | N/A       | N/A     |
        |     └─Selection_31(Probe)              | 820510720.00  | 1025638257 | root      |                                                            | time:1m42.8s, loops:1001601                                                                                                                                                                                                                                                                                                                                                                  | or(eq(sz_online_com.aml_parameter_red_namelist.aml_red_namelist_usno, "8069443"), isnull(sz_online_com.aml_parameter_red_namelist.aml_red_namelist_cusno)) | 64.7 KB   | N/A     |
        |       └─HashJoin_32                    | 1025638400.00 | 1025638400 | root      |                                                            | time:864.5ms, loops:1001604, build_hash_table:{total:994.7µs, fetch:936.4µs, build:58.3µs}, probe:{concurrency:5, total:8m51.2s, max:1m46.2s, probe:8m46.5s, fetch:4.71s}                                                                                                                                                                                                                    | left outer join, equal:[eq(sz_online_com.dwd_com_acc_query.cust_no, sz_online_com.aml_parameter_red_namelist.aml_red_namelist_cusno)]                      | 35.7 KB   | 0 Bytes |
        |         ├─TableReader_40(Build)        | 156.84        | 157        | root      |                                                            | time:909.7µs, loops:2, cop_task: {num: 1, max: 1.19ms, proc_keys: 157, rpc_num: 1, rpc_time: 1.18ms, copr_cache_hit_ratio: 0.00, distsql_concurrency: 15}                                                                                                                                                                                                                                    | data:Selection_39                                                                                                                                          | 5.41 KB   | N/A     |
        |         │ └─Selection_39               | 156.84        | 157        | cop[tikv] |                                                            | tikv_task:{time:4ms, loops:3}, scan_detail: {total_process_keys: 157, total_process_keys_size: 8792, total_keys: 158, get_snapshot_time: 336.4µs, rocksdb: {key_skipped_count: 157, block: {cache_hit_count: 10}}}                                                                                                                                                                           | not(isnull(sz_online_com.aml_parameter_red_namelist.aml_red_namelist_cusno))                                                                               | N/A       | N/A     |
        |         │   └─TableFullScan_38         | 157.00        | 157        | cop[tikv] | table:redNamelist                                          | tikv_task:{time:4ms, loops:3}                                                                                                                                                                                                                                                                                                                                                                | keep order:false                                                                                                                                           | N/A       | N/A     |
        |         └─IndexReader_37(Probe)        | 1025638400.00 | 1025638400 | root      |                                                            | time:1.89s, loops:1003178, cop_task: {num: 28475, max: 100.7ms, min: 648.7µs, avg: 21.3ms, p95: 33ms, max_proc_keys: 50144, p95_proc_keys: 50144, tot_proc: 8m44.6s, tot_wait: 1.68s, rpc_num: 28475, rpc_time: 10m6.3s, copr_cache_hit_ratio: 0.00, distsql_concurrency: 15}                                                                                                                | index:IndexFullScan_36                                                                                                                                     | 27.6 MB   | N/A     |
        |           └─IndexFullScan_36           | 1025638400.00 | 1025638400 | cop[tikv] | table:acc, index:com_idx_acct_no_cust_no(ACCT_NO, CUST_NO) | tikv_task:{proc max:92ms, min:0s, avg: 17.1ms, p80:24ms, p95:28ms, iters:1115162, tasks:28475}, scan_detail: {total_process_keys: 1025638400, total_process_keys_size: 116922777600, total_keys: 1025666875, get_snapshot_time: 2.74s, rocksdb: {key_skipped_count: 1025638400, block: {cache_hit_count: 2088010, read_count: 54, read_byte: 348.9 KB, read_time: 200.4µs}}}                 | keep order:false                                                                                                                                           | N/A       | N/A     |
        +----------------------------------------+---------------+------------+-----------+------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+---------+
        13 rows in set (1 min 46.438 sec)
    """
    tunner = SqlTunner()
    output = tunner.tune("gpt-3.5-turbo-16k", original_sql,
                         schemas, original_plan)
    # output = tunner.tune("gpt-4", schemas, original_sql, original_plan)
    print(output)
