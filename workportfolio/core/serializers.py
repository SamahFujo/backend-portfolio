from rest_framework import serializers


class StartProjectRequestSerializer(serializers.Serializer):
    projectName = serializers.CharField(min_length=2, max_length=200)
    projectType = serializers.CharField(required=False, allow_blank=True, max_length=100)
    budgetRange = serializers.CharField(required=False, allow_blank=True, max_length=100)
    timeline = serializers.CharField(required=False, allow_blank=True, max_length=100)
    projectDescription = serializers.CharField(min_length=20)
    yourName = serializers.CharField(min_length=2, max_length=120)
    yourEmail = serializers.EmailField()