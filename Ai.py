# main.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import tensorflow as tf
import cv2
import pytesseract
from datetime import datetime

class MicroFinanceAI:
    def __init__(self):
        """Initialize models and scalers"""
        self.credit_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def process_loan_application(self, application_data):
        """Main function to process a loan application"""
        try:
            # 1. Process KYC
            kyc_status = self.verify_kyc(application_data.get('kyc_documents', {}))
            if not kyc_status['verified']:
                return {
                    'status': 'rejected',
                    'reason': 'KYC verification failed',
                    'details': kyc_status['errors']
                }
            
            # 2. Calculate Credit Score
            credit_score = self.calculate_credit_score(application_data)
            
            # 3. Process ESG
            esg_score = self.calculate_esg_score(application_data.get('business_data', {}))
            
            # 4. Social Media Analysis
            social_score = self.analyze_social_media(application_data.get('social_data', {}))
            
            # 5. Final Decision
            decision = self.make_loan_decision(credit_score, esg_score, social_score)
            
            return decision
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing application: {str(e)}'
            }

    def verify_kyc(self, documents):
        """Verify KYC documents"""
        verification = {
            'verified': False,
            'errors': [],
            'details': {}
        }
        
        required_docs = ['id_proof', 'address_proof', 'income_proof']
        
        try:
            # Check for required documents
            for doc in required_docs:
                if doc not in documents:
                    verification['errors'].append(f'Missing {doc}')
                    continue
                
                # Verify document
                if doc == 'id_proof':
                    verification['details']['id'] = self._verify_id_document(documents[doc])
                elif doc == 'address_proof':
                    verification['details']['address'] = self._verify_address_document(documents[doc])
                elif doc == 'income_proof':
                    verification['details']['income'] = self._verify_income_document(documents[doc])
            
            # Set verification status
            verification['verified'] = len(verification['errors']) == 0
            
        except Exception as e:
            verification['errors'].append(f'Verification error: {str(e)}')
            
        return verification

    def calculate_credit_score(self, data):
        """Calculate credit score based on various factors"""
        try:
            # Traditional factors
            traditional_score = self._evaluate_traditional_factors(data)
            
            # Bank statements analysis
            bank_score = self._analyze_bank_statements(data.get('bank_statements', {}))
            
            # Business performance
            business_score = self._evaluate_business_performance(data.get('business_data', {}))
            
            # Weights for different components
            weights = {
                'traditional': 0.4,
                'bank': 0.3,
                'business': 0.3
            }
            
            # Calculate final score
            final_score = (
                traditional_score * weights['traditional'] +
                bank_score * weights['bank'] +
                business_score * weights['business']
            )
            
            return {
                'score': final_score,
                'components': {
                    'traditional_score': traditional_score,
                    'bank_score': bank_score,
                    'business_score': business_score
                }
            }
            
        except Exception as e:
            return {
                'score': 0,
                'error': str(e)
            }

    def calculate_esg_score(self, business_data):
        """Calculate ESG score"""
        esg_score = {
            'environmental': 0,
            'social': 0,
            'governance': 0,
            'total': 0,
            'recommendations': []
        }
        
        try:
            # Environmental factors
            environmental_factors = self._assess_environmental_impact(business_data)
            esg_score['environmental'] = environmental_factors['score']
            
            # Social factors
            social_factors = self._assess_social_impact(business_data)
            esg_score['social'] = social_factors['score']
            
            # Governance factors
            governance_factors = self._assess_governance(business_data)
            esg_score['governance'] = governance_factors['score']
            
            # Calculate total score
            esg_score['total'] = (
                esg_score['environmental'] +
                esg_score['social'] +
                esg_score['governance']
            ) / 3
            
            # Add recommendations
            if esg_score['environmental'] < 0.6:
                esg_score['recommendations'].append("Improve environmental practices")
            if esg_score['social'] < 0.6:
                esg_score['recommendations'].append("Enhance social responsibility")
            if esg_score['governance'] < 0.6:
                esg_score['recommendations'].append("Strengthen governance structures")
                
        except Exception as e:
            esg_score['error'] = str(e)
            
        return esg_score

    def analyze_social_media(self, social_data):
        """Analyze social media data"""
        analysis = {
            'sentiment_score': 0,
            'business_activity': 0,
            'risk_factors': [],
            'opportunities': []
        }
        
        try:
            if 'posts' in social_data:
                # Sentiment analysis
                sentiments = []
                for post in social_data['posts']:
                    blob = TextBlob(post)
                    sentiments.append(blob.sentiment.polarity)
                
                analysis['sentiment_score'] = np.mean(sentiments)
                
                # Business activity analysis
                business_keywords = ['business', 'customer', 'product', 'service', 'growth']
                business_mentions = sum(
                    1 for post in social_data['posts']
                    if any(keyword in post.lower() for keyword in business_keywords)
                )
                analysis['business_activity'] = business_mentions / len(social_data['posts'])
                
                # Risk assessment
                if analysis['sentiment_score'] < -0.2:
                    analysis['risk_factors'].append("Negative social media sentiment")
                if analysis['business_activity'] < 0.3:
                    analysis['risk_factors'].append("Low business activity")
                    
                # Opportunity assessment
                if analysis['sentiment_score'] > 0.6:
                    analysis['opportunities'].append("Strong positive online presence")
                if analysis['business_activity'] > 0.7:
                    analysis['opportunities'].append("High business engagement")
                    
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis

    def make_loan_decision(self, credit_score, esg_score, social_score):
        """Make final loan decision"""
        decision = {
            'status': 'pending',
            'loan_amount': 0,
            'interest_rate': 0,
            'term_months': 0,
            'requirements': [],
            'recommendations': []
        }
        
        try:
            # Calculate final score
            final_score = (
                credit_score['score'] * 0.5 +
                esg_score['total'] * 0.3 +
                social_score['sentiment_score'] * 0.2
            )
            
            # Set loan terms based on score
            if final_score >= 0.7:
                decision.update({
                    'status': 'approved',
                    'loan_amount': 500000,
                    'interest_rate': 12,
                    'term_months': 24
                })
            elif final_score >= 0.5:
                decision.update({
                    'status': 'approved',
                    'loan_amount': 300000,
                    'interest_rate': 15,
                    'term_months': 18,
                    'requirements': ['Monthly business review']
                })
            elif final_score >= 0.3:
                decision.update({
                    'status': 'approved',
                    'loan_amount': 100000,
                    'interest_rate': 18,
                    'term_months': 12,
                    'requirements': ['Collateral', 'Monthly business review']
                })
            else:
                decision.update({
                    'status': 'rejected',
                    'reason': 'Low credit score'
                })
                
            # Add recommendations
            decision['recommendations'].extend(esg_score['recommendations'])
            
        except Exception as e:
            decision['status'] = 'error'
            decision['error'] = str(e)
            
        return decision

# Example usage
def main():
    # Initialize the system
    microfinance = MicroFinanceAI()
    
    # Sample application data
    application = {
        'kyc_documents': {
            'id_proof': 'path_to_id_image',
            'address_proof': 'path_to_address_image',
            'income_proof': 'path_to_income_docs'
        },
        'personal_data': {
            'name': 'John Doe',
            'age': 35,
            'income': 50000,
            'expenses': 30000
        },
        'business_data': {
            'type': 'Retail Store',
            'age': 5,
            'revenue': 200000,
            'employees': 3,
            'environmental_practices': ['waste_recycling', 'energy_efficient'],
            'social_initiatives': ['local_employment', 'community_support']
        },
        'bank_statements': {
            'average_balance': 75000,
            'monthly_transactions': 45,
            'bounced_checks': 0
        },
        'social_data': {
            'posts': [
                "Excited to expand my business!",
                "Great customer feedback today",
                "New inventory arriving next week"
            ]
        }
    }
    
    # Process application
    result = microfinance.process_loan_application(application)
    
    # Print results
    print("\nLoan Application Result:")
    print(f"Status: {result['status']}")
    if result['status'] == 'approved':
        print(f"Loan Amount: ₹{result['loan_amount']:,}")
        print(f"Interest Rate: {result['interest_rate']}%")
        print(f"Term: {result['term_months']} months")
        if result['requirements']:
            print("\nRequirements:")
            for req in result['requirements']:
                print(f"- {req}")
    elif result['status'] == 'rejected':
        print(f"Reason: {result.get('reason', 'Unknown')}")
    
    if result.get('recommendations'):
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    main()