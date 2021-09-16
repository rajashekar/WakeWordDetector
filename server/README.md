
### Elastic Beanstalk initialize app
```
eb init -p docker-19.03.13-ce wakebot-app --region us-west-2
```

### Create Elastic Beanstalk instance
```
eb create wakebot-app --instance_type t2.large --max-instances 1
```