#!/usr/bin/env python3
"""
CLI Tool for Autonomous Insurance Credentialing System
Manage calls, view metrics, and interact with the system
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Optional
import requests
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

# Import from credentialing agent
from credentialing_agent import DatabaseManager

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000/api")


class CredentialingCLI:
    """Command-line interface for the credentialing system"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api_url = API_BASE_URL
    
    def start_call(self, args):
        """Start a new credentialing call"""
        # Load from file or command line
        if args.file:
            with open(args.file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'insurance_name': args.insurance,
                'provider_name': args.provider,
                'npi': args.npi,
                'tax_id': args.tax_id,
                'address': args.address,
                'insurance_phone': args.phone,
                'questions': args.questions.split(',') if args.questions else []
            }
        
        print(f"Starting call for {data['provider_name']} → {data['insurance_name']}...")
        
        try:
            response = requests.post(
                f"{self.api_url}/start-call",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Call initiated successfully")
            print(f"  Provider: {result.get('provider')}")
            print(f"  Insurance: {result.get('insurance')}")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error starting call: {e}")
            sys.exit(1)
    
    def show_status(self, args):
        """Show status of a specific call"""
        try:
            response = requests.get(
                f"{self.api_url}/call-status/{args.call_id}",
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success'):
                print(f"✗ Call not found: {args.call_id}")
                return
            
            print(f"\n{'='*60}")
            print(f"Call Status: {args.call_id}")
            print(f"{'='*60}")
            print(f"Provider: {result.get('provider_name')}")
            print(f"Insurance: {result.get('insurance_name')}")
            print(f"Call State: {result.get('call_state')}")
            print(f"Status: {result.get('status', 'In progress')}")
            print(f"Reference #: {result.get('reference_number', 'N/A')}")
            print(f"Notes: {result.get('notes', 'None')}")
            print(f"{'='*60}\n")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching status: {e}")
    
    def show_transcript(self, args):
        """Show full conversation transcript"""
        try:
            response = requests.get(
                f"{self.api_url}/call-transcript/{args.call_id}",
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success'):
                print(f"✗ Call not found: {args.call_id}")
                return
            
            conversation = result.get('conversation', [])
            
            print(f"\n{'='*60}")
            print(f"Conversation Transcript: {args.call_id}")
            print(f"{'='*60}\n")
            
            for msg in conversation:
                speaker = msg['speaker'].upper()
                message = msg['message']
                print(f"{speaker}: {message}\n")
            
            print(f"{'='*60}\n")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching transcript: {e}")
    
    def show_metrics(self, args):
        """Show system metrics"""
        try:
            response = requests.get(
                f"{self.api_url}/metrics",
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"\n{'='*60}")
            print(f"System Metrics (Last {result.get('period_days', 7)} days)")
            print(f"{'='*60}")
            print(f"Total Calls: {result.get('total_calls', 0)}")
            print(f"Approved: {result.get('approved', 0)}")
            print(f"In Progress: {result.get('in_progress', 0)}")
            print(f"Success Rate: {result.get('success_rate', 0):.2f}%")
            print(f"{'='*60}\n")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching metrics: {e}")
    
    def list_calls(self, args):
        """List recent calls from database"""
        with self.db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    insurance_name,
                    provider_name,
                    status,
                    reference_number,
                    created_at
                FROM credentialing_requests
                ORDER BY created_at DESC
                LIMIT %s
            """, (args.limit,))
            
            rows = cur.fetchall()
        
        if not rows:
            print("No calls found.")
            return
        
        table_data = []
        for row in rows:
            table_data.append([
                str(row[0])[:8] + "...",  # ID
                row[1],  # Insurance
                row[2],  # Provider
                row[3],  # Status
                row[4] or "N/A",  # Ref #
                row[5].strftime("%Y-%m-%d %H:%M")  # Created
            ])
        
        headers = ["ID", "Insurance", "Provider", "Status", "Ref #", "Created"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def list_followups(self, args):
        """List pending follow-ups"""
        try:
            response = requests.get(
                f"{self.api_url}/scheduled-followups",
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            followups = result.get('followups', [])
            
            if not followups:
                print("No pending follow-ups.")
                return
            
            table_data = []
            for f in followups:
                table_data.append([
                    f['insurance_name'],
                    f['provider_name'],
                    f['action_type'],
                    f['scheduled_date'][:10] if f['scheduled_date'] else "N/A"
                ])
            
            headers = ["Insurance", "Provider", "Action", "Scheduled"]
            print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching follow-ups: {e}")
    
    def add_ivr_knowledge(self, args):
        """Add new IVR knowledge"""
        data = {
            'insurance_name': args.insurance,
            'menu_level': args.level,
            'detected_phrase': args.phrase,
            'preferred_action': args.action,
            'action_value': args.value
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/ivr-knowledge",
                json=data,
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ IVR knowledge added successfully")
            print(f"  ID: {result.get('ivr_id')}")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error adding IVR knowledge: {e}")
    
    def show_ivr_knowledge(self, args):
        """Show IVR knowledge for an insurance provider"""
        with self.db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    menu_level,
                    detected_phrase,
                    preferred_action,
                    action_value,
                    success_rate,
                    attempts
                FROM ivr_knowledge
                WHERE insurance_name ILIKE %s
                ORDER BY menu_level, success_rate DESC
            """, (f"%{args.insurance}%",))
            
            rows = cur.fetchall()
        
        if not rows:
            print(f"No IVR knowledge found for '{args.insurance}'")
            return
        
        table_data = []
        for row in rows:
            table_data.append([
                row[0],  # Level
                row[1][:40] + "..." if len(row[1]) > 40 else row[1],  # Phrase
                row[2],  # Action
                row[3] or "N/A",  # Value
                f"{row[4]*100:.1f}%" if row[4] else "0%",  # Success rate
                row[5]  # Attempts
            ])
        
        headers = ["Level", "Phrase", "Action", "Value", "Success", "Attempts"]
        print(f"\nIVR Knowledge for '{args.insurance}':")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def export_data(self, args):
        """Export call data to JSON"""
        with self.db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    cr.*,
                    json_agg(
                        json_build_object(
                            'speaker', ch.speaker,
                            'message', ch.message,
                            'timestamp', ch.timestamp
                        ) ORDER BY ch.timestamp
                    ) as conversation
                FROM credentialing_requests cr
                LEFT JOIN conversation_history ch ON cr.id::text = ch.call_id
                WHERE cr.created_at > NOW() - INTERVAL '%s days'
                GROUP BY cr.id
            """, (args.days,))
            
            rows = cur.fetchall()
        
        # Convert to dict
        data = []
        for row in rows:
            data.append({
                'id': str(row[0]),
                'insurance_name': row[1],
                'provider_name': row[2],
                'status': row[6],
                'conversation': row[-1]
                # Add more fields as needed
            })
        
        output_file = args.output or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Exported {len(data)} calls to {output_file}")
    
    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Insurance Credentialing System CLI"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start call command
    start_parser = subparsers.add_parser('start', help='Start a new credentialing call')
    start_parser.add_argument('--file', help='JSON file with call details')
    start_parser.add_argument('--insurance', help='Insurance name')
    start_parser.add_argument('--provider', help='Provider name')
    start_parser.add_argument('--npi', help='NPI number')
    start_parser.add_argument('--tax-id', help='Tax ID')
    start_parser.add_argument('--address', help='Provider address')
    start_parser.add_argument('--phone', help='Insurance phone number')
    start_parser.add_argument('--questions', help='Comma-separated questions')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show call status')
    status_parser.add_argument('call_id', help='Call ID')
    
    # Transcript command
    transcript_parser = subparsers.add_parser('transcript', help='Show call transcript')
    transcript_parser.add_argument('call_id', help='Call ID')
    
    # Metrics command
    subparsers.add_parser('metrics', help='Show system metrics')
    
    # List calls command
    list_parser = subparsers.add_parser('list', help='List recent calls')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of calls to show')
    
    # Follow-ups command
    subparsers.add_parser('followups', help='List pending follow-ups')
    
    # Add IVR knowledge
    ivr_add_parser = subparsers.add_parser('add-ivr', help='Add IVR knowledge')
    ivr_add_parser.add_argument('--insurance', required=True, help='Insurance name')
    ivr_add_parser.add_argument('--level', type=int, required=True, help='Menu level')
    ivr_add_parser.add_argument('--phrase', required=True, help='Detected phrase')
    ivr_add_parser.add_argument('--action', required=True, choices=['dtmf', 'speech', 'wait'])
    ivr_add_parser.add_argument('--value', help='Action value (digit or phrase)')
    
    # Show IVR knowledge
    ivr_show_parser = subparsers.add_parser('show-ivr', help='Show IVR knowledge')
    ivr_show_parser.add_argument('insurance', help='Insurance name')
    
    # Export data
    export_parser = subparsers.add_parser('export', help='Export call data')
    export_parser.add_argument('--days', type=int, default=7, help='Days to export')
    export_parser.add_argument('--output', help='Output file name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = CredentialingCLI()
    
    try:
        if args.command == 'start':
            cli.start_call(args)
        elif args.command == 'status':
            cli.show_status(args)
        elif args.command == 'transcript':
            cli.show_transcript(args)
        elif args.command == 'metrics':
            cli.show_metrics(args)
        elif args.command == 'list':
            cli.list_calls(args)
        elif args.command == 'followups':
            cli.list_followups(args)
        elif args.command == 'add-ivr':
            cli.add_ivr_knowledge(args)
        elif args.command == 'show-ivr':
            cli.show_ivr_knowledge(args)
        elif args.command == 'export':
            cli.export_data(args)
    finally:
        cli.close()


if __name__ == '__main__':
    main()
