"""
Tools and utilities to work with

https://github.com/unitedstates/congress


fetch bill metadata to data/118/bills
```
usc-run govinfo --bulkdata=BILLSTATUS --congress=118
```

fetch bill text to data/govinfo/BILLS
```
usc-run govinfo --bulkdata=BILLS --congress=118
```

fetch plaw text to data/govinfo/PLAWS
```
usc-run govinfo --bulkdata=PLAW --congress=118
```

"""
from collections import Counter
import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Optional, Union

from bs4 import BeautifulSoup
from datasets import Dataset
import huggingface_hub
from huggingface_hub import HfApi
from langchain.schema import Document
import pandas as pd
from pydantic import BaseModel
import xmltodict

from hyperdemocracy import langchain_helpers


logger = logging.getLogger(__name__)


MIN_DATETIME = datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(0)))
CONGRES_GOV_TYPE_MAP = {
    "HCONRES": "house-concurrent-resolution",
    "HJRES": "house-joint-resolution",
    "HR": "house-bill",
    "HRES": "house-resolution",
    "S": "senate-bill",
    "SCONRES": "senate-concurrent-resolution",
    "SJRES": "senate-joint-resolution",
    "SRES": "senate-resolution",
}


class Activity(BaseModel):
    name: str
    date: datetime.datetime


class TextVersion(BaseModel):
    type: str
    date: Optional[datetime.datetime] = None
    url: Optional[str] = None
    #formats: list[str]


class CboCostEstimate(BaseModel):
    pubDate: datetime.datetime
    title: str
    url: str
    description: str


class Summary(BaseModel):
    versionCode: str
    actionDate: datetime.date
    actionDesc: str
    updateDate: datetime.datetime
    text: str


class Sponsor(BaseModel):
    bioguideId: str
    fullName: str
    firstName: str
    lastName: str
    party: str
    state: str
    middleName: Optional[str] = None
    isByRequest: str


class Cosponsor(BaseModel):
    bioguideId: str
    fullName: str
    firstName: str
    lastName: str
    party: str
    state: str
    middleName: Optional[str] = None
    sponsorshipDate: datetime.date
    isOriginalCosponsor: bool


class Subcommittee(BaseModel):
    systemCode: str
    name: str
    activities: list[Activity]


class Committee(BaseModel):
    systemCode: str
    name: str
    chamber: Optional[str] = None
    type: Optional[str] = None
    subcommittees: list[Subcommittee] = []
    activities: list[Activity] = []


class Action(BaseModel):
    actionDate: datetime.date
    text: str
    type: Optional[str] = None
    actionCode: Optional[str] = None
    sourceSystemCode: Optional[str] = None
    sourceSystemName: Optional[str] = None
    committees: list[Committee] = []


class RelationshipDetail(BaseModel):
    type: str
    identifiedBy: str


class RelatedBill(BaseModel):
    title: str
    congress: int
    number: int
    type: str
    latestActionDate: datetime.date
    latestActionText: str
    relationshipDetails: list[RelationshipDetail]


class Title(BaseModel):
    titleType: str
    title: str
    billTextVersionName: Optional[str] = None
    billTextVersionCode: Optional[str] = None


class Bill(BaseModel):
    number: int
    updateDate: datetime.datetime
    updateDateIncludingText: datetime.datetime
    originChamber: str
    type: str
    introducedDate: datetime.date
    congress: int
    constitutionalAuthorityStatementText: Optional[str] = None
    sponsors: list[Sponsor]
    cosponsors: list[Cosponsor]
    committees: list[Committee]
    relatedBills: list[RelatedBill]
    actions: list[Action]
    cboCostEstimates: list[CboCostEstimate]
    # cdata ?
    subjects: list[str] = []
    policyArea: Optional[str] = None
    summaries: list[Summary] = []
    latestAction: Action
    one_title: str
    titles: list[Title]
    textVersions: list[TextVersion]


class BillMetadataXml:

    def __init__(self, xml: str):
        self.xml = xml
        self.d = xmltodict.parse(xml)
        self.d_bill = self.d["billStatus"]["bill"]

    def fetch_item_list(self, sub, key):
        val = sub.get(key)
        if val is None:
            return []
        else:
            x = val["item"]
            if isinstance(x, dict):
                x = [x]
            return x

    def sponsors(self):
        return [Sponsor(**el) for el in self.fetch_item_list(self.d_bill, "sponsors")]

    def cosponsors(self):
        return [Cosponsor(**el) for el in self.fetch_item_list(self.d_bill, "cosponsors")]

    def subcommittees(self, d_committee):
        subcomms = []
        for d_subcomm in self.fetch_item_list(d_committee, "subcommittees"):
            acts = [Activity(**d_act) for d_act in self.fetch_item_list(d_subcomm, "activities")]
            subcomm = Subcommittee(
                systemCode = d_subcomm["systemCode"],
                name = d_subcomm["name"],
                activities = acts,
            )
            subcomms.append(subcomm)
        return subcomms

    def committees(self):
        comms = []
        for d_comm in self.fetch_item_list(self.d_bill, "committees"):
            acts = [Activity(**d_act) for d_act in self.fetch_item_list(d_comm, "activities")]
            subcomms = self.subcommittees(d_comm)
            comm = Committee(
                systemCode = d_comm["systemCode"],
                name = d_comm["name"],
                chamber = d_comm["chamber"],
                type = d_comm["type"],
                subcommittees = subcomms,
                activities = acts,
            )
            comms.append(comm)
        return comms

    def related_bills(self):
        rbills = []
        for d_rbill in self.fetch_item_list(self.d_bill, "relatedBills"):
            rdeets = [
                RelationshipDetail(**d_rdeet)
                for d_rdeet in self.fetch_item_list(d_rbill, "relationshipDetails")
            ]
            rbill = RelatedBill(
                title = d_rbill["title"],
                congress = d_rbill["congress"],
                number = d_rbill["number"],
                type = d_rbill["type"],
                latestActionDate = d_rbill["latestAction"]["actionDate"],
                latestActionText = d_rbill["latestAction"]["text"],
                relationshipDetails = rdeets,
            )
            rbills.append(rbill)
        return rbills

    def actions(self):
        actions = []
        for d_action in self.fetch_item_list(self.d_bill, "actions"):
            comms = [
                Committee(**d_comm)
                for d_comm in self.fetch_item_list(d_action, "committees")
            ]
            action = Action(
                actionDate = d_action["actionDate"],
                text = d_action["text"],
                type = d_action.get("type"),
                actionCode = d_action.get("actionCode"),
                sourceSystemCode = d_action["sourceSystem"].get("code"),
                sourceSystemName = d_action["sourceSystem"].get("name"),
                committees = comms,
            )
            actions.append(action)
        return actions

    def cbo_cost_estimates(self):
        return [CboCostEstimate(**el) for el in self.fetch_item_list(self.d_bill, "cboCostEstimates")]

    def subjects(self):
        """This one does not follow the pattern"""
        val = self.d_bill.get("subjects")
        if val is None:
            return []
        else:
            return [
                el["name"]
                for el in self.fetch_item_list(self.d_bill["subjects"], "legislativeSubjects")
            ]

    def policy_area(self):
        val = self.d_bill.get("policyArea")
        if val is None:
            return None
        else:
            return self.d_bill["policyArea"]["name"]

    def summaries(self):
        """This one does not follow the pattern

        # TODO: sort by updateDate or actionDate?
        """
        val = self.d_bill.get("summaries")
        if val is None:
            summs = []
        else:
            l_of_d = val["summary"]
            if isinstance(l_of_d, dict):
                l_of_d = [l_of_d]
            summs = [Summary(**el) for el in l_of_d]

        summs = sorted(summs, key=lambda x: x.actionDate)
        return summs

    def latest_action(self):
        return Action(**self.d_bill["latestAction"])

    def one_title(self):
        return self.d_bill["title"]

    def titles(self):
        return [Title(**el) for el in self.fetch_item_list(self.d_bill, "titles")]

    def text_versions(self):
        val = self.d_bill.get("textVersions")
        if val is None:
            text_vers = []

        else:
            text_vers = []
            tvs = val["item"]
            if isinstance(tvs, dict):
                tvs = [tvs]
            for d_tv in tvs:
                url = d_tv.get("formats")
                if url is not None:
                    url = url.get("item")
                    if url is not None:
                        url = url.get("url")
                tv = TextVersion(
                    type = d_tv["type"],
                    date = d_tv.get("date"),
                    url = url,
                )
                text_vers.append(tv)

        # sort in chronological order
        text_vers = sorted(text_vers, key=lambda x: x.date or MIN_DATETIME)
        return text_vers


    def as_pydantic(self):
        return Bill(
            number=self.d_bill["number"],
            updateDate=self.d_bill["updateDate"],
            updateDateIncludingText=self.d_bill["updateDateIncludingText"],
            originChamber=self.d_bill["originChamber"],
            type=self.d_bill["type"],
            introducedDate=self.d_bill["introducedDate"],
            congress=self.d_bill["congress"],
            constitutionalAuthorityStatementText=self.d_bill.get("constitutionalAuthorityStatementText"),
            sponsors=self.sponsors(),
            cosponsors=self.cosponsors(),
            committees=self.committees(),
            relatedBills=self.related_bills(),
            actions=self.actions(),
            cboCostEstimates=self.cbo_cost_estimates(),
            subjects=self.subjects(),
            policyArea=self.policy_area(),
            summaries=self.summaries(),
            latestAction=self.latest_action(),
            one_title=self.one_title(),
            titles=self.titles(),
            textVersions=self.text_versions(),
        )



def get_hf_row(metadata_path: Union[str, Path], base_text_path: Union[str, Path]):

    with open(metadata_path, "r") as fp:
        metadata_xml = fp.read()
    bmx = BillMetadataXml(metadata_xml)
    bill = bmx.as_pydantic()

    # for now the huggingface datasets preview doesn't suport datetimes
    # i'm going to keep them as strings b/c people can also reconvert them

    bill_dict_dt = bill.dict() # this will keep datetime objects
    bill_dict = json.loads(bill.json())  # this will make the datetime objects strings

    congress_gov_url = "https://www.congress.gov/bill/{}th-congress/{}/{}".format(
        bill_dict["congress"],
        CONGRES_GOV_TYPE_MAP[bill_dict["type"]],
        bill_dict["number"],
    )
    govtrack_url = "https://www.govtrack.us/congress/bills/{}/{}{}".format(
        bill_dict["congress"],
        bill_dict["type"].lower(),
        bill_dict["number"],
    )

    # take most recent summary
    # TODO: check if this matches the text version
    if len(bill_dict["summaries"]) == 0:
        summary_text = pd.NA
        summary_meta = pd.NA
    else:
        summary = bill_dict["summaries"][-1]
        summary_text = summary.pop("text")
        summary_meta = summary

    base_row = {
        "id": "{}-{}-{}".format(bill_dict["congress"], bill_dict["type"], bill_dict["number"]),
        "title": bill_dict["one_title"],
        "congress": bill_dict["congress"],
        "type": bill_dict["type"],
        "number": bill_dict["number"],
        "origin_chamber": bill_dict["originChamber"],
        "sponsors": bill_dict["sponsors"],
        "cosponsors": bill_dict["cosponsors"],
        "congress_gov_url": congress_gov_url,
        "govtrack_url": govtrack_url,
        "summary_text": summary_text,
        "summary_meta": summary_meta,
        "subjects": bill_dict["subjects"],
        "policy_area": bill_dict["policyArea"],
        "bill": bill_dict,
        "metadata_xml": metadata_xml,
    }

    if len(bill_dict["textVersions"]) == 0:
        # no text versions available
        stats_key = "no_tvs"
        extras_row = {
            "text_type": pd.NA,
            "text_date": pd.NA,
            "text_url": pd.NA,
            "text_xml": pd.NA,
            "text": pd.NA,
        }
        row = {**base_row, **extras_row}
        logger.info(f"{metadata_path} has no text versions. {congress_gov_url}")
        return row, stats_key

    # take the most recent text version
    # TODO: take less recent versions if we dont find url?
    tv = bill_dict["textVersions"][-1]

    if tv["url"] is None:
        # no text url available
        stats_key = "no_url"
        extras_row = {
            "text_type": tv["type"],
            "text_date": tv["date"],
            "text_url": pd.NA,
            "text_xml": pd.NA,
            "text": pd.NA,
        }
        row = {**base_row, **extras_row}
        logger.info(f"{metadata_path} has no text versions with url. {congress_gov_url}")
        return row, stats_key


    tv_fname = Path(tv["url"]).name
    if tv_fname.startswith("BILLS"):
        pattern = "BILLS-(\d{1,3})(hconres|hjres|hr|hres|s|sconres|sjres|sres)(\d{1,5})([A-Za-z]+)\.xml"
        match = re.match(pattern, tv_fname)
        if match is None:
            raise ValueError()
        congres, bill_type, bill_number, bill_version = match.groups()
        text_path = base_text_path / "BILLS" / str(bill_dict["congress"]) / str(1) /  bill_type / tv_fname

    elif tv_fname.startswith("PLAW"):
        pattern = "PLAW-(\d{1,3})(publ)(\d{1,5})\.xml"
        match = re.match(pattern, tv_fname)
        if match is None:
            raise ValueError()
        congres, bill_type, bill_number = match.groups()
        bill_version = None
        text_path = base_text_path / "PLAW" / str(bill_dict["congress"]) / "public" / tv_fname

    else:
        raise ValueError()


    if not text_path.exists():
        # no text file found
        stats_key = "no_file"
        extras_row = {
            "text_type": tv["type"],
            "text_date": tv["date"],
            "text_url": tv["url"],
            "text_xml": pd.NA,
            "text": pd.NA,
        }
        row = {**base_row, **extras_row}
        logger.info(f"{metadata_path} has no local file matching {tv} (checked {text_path}). {congress_gov_url}")
        return row, stats_key


    with text_path.open("r") as fp:
        text_xml = fp.read()

    soup = BeautifulSoup(text_xml, 'xml')
    bill_text = soup.get_text(separator=" ").strip()

    # we can try other things later
    #bill_preamble_text = soup.find("preamble").get_text(separator=" ")
    #bill_resolution_body_text = soup.find("resolution-body").get_text(separator=" ")
    #dd = xmltodict.parse(text_xml)


    stats_key = "ok"
    extras_row = {
        "text_type": tv["type"],
        "text_date": tv["date"],
        "text_url": tv["url"],
        "text_xml": text_xml,
        "text": bill_text,
    }
    row = {**base_row, **extras_row}
    return row, stats_key


def get_hf_dataframe(base_data_path: Union[str, Path], base_congress: int, max_rows: Optional[int]=None):

    logger.info(f"writing hf parquet with {base_data_path=} and {base_congress=}")
    base_text_path = base_data_path / "data" / "govinfo"
    metadata_paths = sorted(list((base_data_path / "data" / str(base_congress) / "bills").glob("*/*/*.xml")))
    logger.info(f"found {len(metadata_paths)} bill metadata files")

    if max_rows is not None and max_rows < len(metadata_paths):
        logger.info(f"restricting file list to {max_rows} entries")
        metadata_paths = metadata_paths[:max_rows]

    stats = {
        "no_tvs": 0,
        "no_url": 0,
        "no_file": 0,
        "ok": 0,
    }

    bill_types = Counter()
    bill_versions = Counter()

    rows = []
    for metadata_path in metadata_paths:
        row, stats_key = get_hf_row(metadata_path, base_text_path)
        stats[stats_key] += 1
        rows.append(row)

    logging.info(f"{stats=}")

    df = pd.DataFrame(rows)
    return df


def get_langchain_docs(hf_dataset: Dataset) -> list[Document]:
    docs = []
    for row in hf_dataset:
        if row["text"] is None:
            continue

        metadata = {
            "id": row["id"],
            "title": row["title"],
            "congress": row["congress"],
            "type": row["type"],
            "number": row["number"],
            "origin_chamber": row["origin_chamber"],
            "sponsors": row["sponsors"],
            "cosponsors": row["cosponsors"],
            "congress_gov_url": row["congress_gov_url"],
            "govtrack_url": row["govtrack_url"],
            "summary_text": row["summary_text"],
            "text_url": row["text_url"],
            "text_type": row["text_type"],
            "text_date": row["text_date"],
            "subjects": row["subjects"],
            "policy_area": row["policy_area"],
        }

        doc = Document(
            page_content=row['text'],
            metadata=metadata,
        )
        docs.append(doc)

    return docs

def upload_file_to_hf(hf_org, dataset_name, local_path, repo_path, dryrun=True):
    """Upload a file to hub dataset repository.

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_file
    """
    repo_id = os.path.join(hf_org, dataset_name)

    print(f"going to upload {local_path} to {repo_id} as {repo_path}")
    if dryrun:
        print("this is a dryrun. to actually upload, run again with dryrun=False")
        return

    api = HfApi()
    print(f"uploading {local_path} to {repo_id}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"upload {local_path} to hub from bigbio repo",
        commit_description=f"upload {local_path} to hub from bigbio repo",
    )



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    base_data_path = Path("/mnt/disks/data2/hyperdemocracy/s3/hyperdemocracy/congress-scraper")
    base_congress = 118
#    max_rows = 10
    max_rows = None

    df = get_hf_dataframe(base_data_path, base_congress, max_rows)
    local_path = f"data-{base_congress}.parquet"
    df.to_parquet(local_path)

    hf_org = "hyperdemocracy"
    hf_dataset_name = "us-congress-bills"
    repo_path = "data" + "/" + local_path
    upload_file_to_hf(hf_org, hf_dataset_name, local_path, repo_path, dryrun=False)
