#!/usr/bin/env python3
"""PulseSlot AI - Adaptive YouTube Posting Intelligence Engine."""

import sys
import argparse


def main():
    """Main entry point with helpful commands."""
    parser = argparse.ArgumentParser(
        description='PulseSlot AI - Optimize YouTube posting times',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore the dataset
  python main.py explore --country US
  
  # Train the model
  python main.py train --countries US GB --sample_size 5000
  
  # Generate posting schedule
  python main.py schedule
  
  # Initialize database
  python main.py init-db
  
For more options, run:
  python scripts/explore_dataset.py --help
  python scripts/train_model.py --help
  python scripts/generate_schedule.py --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Explore dataset')
    explore_parser.add_argument('--country', help='Country code to analyze')
    explore_parser.add_argument('--sample', type=int, help='Number of sample rows to show')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--countries', nargs='+', help='Countries to use')
    train_parser.add_argument('--sample_size', type=int, help='Sample size for testing')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Generate posting schedule')
    
    # Init DB command
    initdb_parser = subparsers.add_parser('init-db', help='Initialize database')
    initdb_parser.add_argument('--load-dataset', action='store_true', 
                              help='Load dataset into database')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test dataset loading')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute commands
    if args.command == 'explore':
        cmd = ['python', 'scripts/explore_dataset.py']
        if args.country:
            cmd.extend(['--country', args.country])
        if args.sample:
            cmd.extend(['--sample', str(args.sample)])
        import subprocess
        return subprocess.call(cmd)
    
    elif args.command == 'train':
        cmd = ['python', 'scripts/train_model.py']
        if args.countries:
            cmd.extend(['--countries'] + args.countries)
        if args.sample_size:
            cmd.extend(['--sample_size', str(args.sample_size)])
        import subprocess
        return subprocess.call(cmd)
    
    elif args.command == 'schedule':
        import subprocess
        return subprocess.call(['python', 'scripts/generate_schedule.py'])
    
    elif args.command == 'init-db':
        cmd = ['python', 'scripts/init_db.py']
        if args.load_dataset:
            cmd.append('--load-dataset')
        import subprocess
        return subprocess.call(cmd)
    
    elif args.command == 'test':
        import subprocess
        return subprocess.call(['python', 'test_dataset.py'])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
