import requests
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import time
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityDataCollector:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
        
    def collect_nvd_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Collect data from NIST National Vulnerability Database."""
        logger.info("Collecting data from NVD...")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {
            "pubStartDate": start_date,
            "pubEndDate": end_date,
            "resultsPerPage": 2000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for vuln in data.get("vulnerabilities", []):
                cve = vuln.get("cve", {})
                description = cve.get("descriptions", [{}])[0].get("value", "")
                
                qa_pair = {
                    "instruction": f"What is the vulnerability {cve.get('id')}?",
                    "input": "",
                    "output": description
                }
                self.data.append(qa_pair)
                
        except Exception as e:
            logger.error(f"Error collecting NVD data: {e}")
            
        return self.data

    def collect_mitre_attack(self) -> List[Dict]:
        """Collect data from MITRE ATT&CK framework."""
        logger.info("Collecting data from MITRE ATT&CK...")
        
        url = "https://attack.mitre.org/enterprise/enterprise.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            for technique in data.get("objects", []):
                if technique.get("type") == "attack-pattern":
                    name = technique.get("name", "")
                    description = technique.get("description", "")
                    
                    qa_pair = {
                        "instruction": f"What is the MITRE ATT&CK technique {name}?",
                        "input": "",
                        "output": description
                    }
                    self.data.append(qa_pair)
                    
        except Exception as e:
            logger.error(f"Error collecting MITRE ATT&CK data: {e}")
            
        return self.data

    def collect_owasp_data(self) -> List[Dict]:
        """Collect data from OWASP documentation."""
        logger.info("Collecting data from OWASP...")
        
        # OWASP Top 10 2021
        owasp_top10 = {
            "A01:2021 – Broken Access Control": "Broken access control allows attackers to bypass authorization and perform tasks as though they were privileged users.",
            "A02:2021 – Cryptographic Failures": "Failures related to cryptography which often lead to sensitive data exposure or system compromise.",
            "A03:2021 – Injection": "Injection flaws allow attackers to relay malicious code through an application to another system.",
            "A04:2021 – Insecure Design": "Insecure design is a broad category representing different weaknesses, expressed as 'missing or ineffective control design'.",
            "A05:2021 – Security Misconfiguration": "Security misconfiguration is the most commonly seen issue, often due to insecure default configurations.",
            "A06:2021 – Vulnerable and Outdated Components": "Using components with known vulnerabilities can lead to serious security issues.",
            "A07:2021 – Identification and Authentication Failures": "Authentication and session management functions are often implemented incorrectly.",
            "A08:2021 – Software and Data Integrity Failures": "Software and data integrity failures relate to code and infrastructure that does not protect against integrity violations.",
            "A09:2021 – Security Logging and Monitoring Failures": "Insufficient logging and monitoring, coupled with missing or ineffective integration with incident response.",
            "A10:2021 – Server-Side Request Forgery": "SSRF flaws occur when a web application is fetching a remote resource without validating the user-supplied URL."
        }
        
        for category, description in owasp_top10.items():
            qa_pair = {
                "instruction": f"What is {category} in OWASP Top 10?",
                "input": "",
                "output": description
            }
            self.data.append(qa_pair)
            
        return self.data

    def collect_stackoverflow_security(self, max_pages: int = 5) -> List[Dict]:
        """Collect security-related questions from Stack Overflow."""
        logger.info("Collecting data from Stack Overflow...")
        
        base_url = "https://api.stackexchange.com/2.3/questions"
        params = {
            "tagged": "security",
            "site": "stackoverflow",
            "sort": "votes",
            "order": "desc",
            "pagesize": 100
        }
        
        for page in range(1, max_pages + 1):
            try:
                params["page"] = page
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for question in data.get("items", []):
                    title = question.get("title", "")
                    body = question.get("body", "")
                    # Clean HTML from body
                    body = BeautifulSoup(body, "html.parser").get_text()
                    
                    qa_pair = {
                        "instruction": title,
                        "input": "",
                        "output": body
                    }
                    self.data.append(qa_pair)
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting Stack Overflow data: {e}")
                
        return self.data

    def save_data(self, filename: str = "security_dataset.csv"):
        """Save collected data to CSV file."""
        if not self.data:
            logger.warning("No data to save!")
            return
            
        df = pd.DataFrame(self.data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(self.data)} entries to {output_path}")

def main():
    collector = SecurityDataCollector()
    
    # Collect data from various sources
    collector.collect_nvd_data()
    collector.collect_mitre_attack()
    collector.collect_owasp_data()
    collector.collect_stackoverflow_security()
    
    # Save collected data
    collector.save_data()
    
    logger.info("Data collection completed!")

if __name__ == "__main__":
    main() 