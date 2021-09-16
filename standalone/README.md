
### Elastic Beanstalk initialize app
```
eb init -p python-3.7 wakebot-std-app --region us-west-2
```

### Create Elastic Beanstalk instance
```
eb create wakebot-std-app --instance_type t2.large --max-instances 1
```