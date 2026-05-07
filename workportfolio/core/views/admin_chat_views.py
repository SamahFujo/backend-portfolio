"""
Admin chatbot API views.

These APIs are used by the admin panel to review chatbot sessions,
messages, and chatbot statistics.
"""

from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework import status

from core.models import ChatSession, ChatMessage
from core.permissions import HasInternalAPIKey
from core.serializers import (
    AdminChatSessionListSerializer,
    AdminChatSessionDetailSerializer,
)
from django.db.models import Q


class AdminChatSessionListAPIView(ListAPIView):
    """
    List all chatbot sessions for the admin panel.

    Supports filtering by:
    - visitor_id
    - visitor_email
    - is_active
    - search
    """

    serializer_class = AdminChatSessionListSerializer
    permission_classes = [HasInternalAPIKey]

    def get_queryset(self):
        queryset = (
            ChatSession.objects
            .all()
            .prefetch_related("messages")
            .order_by("-updated_at")
        )

        visitor_id = self.request.query_params.get("visitor_id")
        visitor_email = self.request.query_params.get("visitor_email")
        is_active = self.request.query_params.get("is_active")
        search = self.request.query_params.get("search")

        if visitor_id:
            queryset = queryset.filter(visitor_id__icontains=visitor_id)

        if visitor_email:
            queryset = queryset.filter(visitor_email__icontains=visitor_email)

        if is_active in ["true", "false"]:
            queryset = queryset.filter(is_active=is_active == "true")

        if search:
            queryset = queryset.filter(
                Q(visitor_email__icontains=search) |
                Q(visitor_id__icontains=search) |
                Q(ip_address__icontains=search) |
                Q(messages__content__icontains=search)
            ).distinct()

        return queryset


class AdminChatSessionDetailAPIView(RetrieveAPIView):
    """
    View one full chatbot conversation session.
    """

    serializer_class = AdminChatSessionDetailSerializer
    permission_classes = [HasInternalAPIKey]
    lookup_field = "id"
    lookup_url_kwarg = "session_id"

    def get_queryset(self):
        return (
            ChatSession.objects
            .all()
            .prefetch_related("messages")
        )


class AdminChatStatsAPIView(APIView):
    """
    Show chatbot analytics summary for the admin dashboard.
    """

    permission_classes = [HasInternalAPIKey]

    def get(self, request, *args, **kwargs):
        total_sessions = ChatSession.objects.count()
        active_sessions = ChatSession.objects.filter(is_active=True).count()
        total_messages = ChatMessage.objects.count()
        user_messages = ChatMessage.objects.filter(role="user").count()
        assistant_messages = ChatMessage.objects.filter(
            role="assistant").count()

        unique_visitors = (
            ChatSession.objects
            .exclude(visitor_id__isnull=True)
            .exclude(visitor_id="")
            .values("visitor_id")
            .distinct()
            .count()
        )

        latest_sessions = ChatSession.objects.order_by("-updated_at")[:5]

        latest_sessions_data = AdminChatSessionListSerializer(
            latest_sessions,
            many=True,
        ).data

        return Response(
            {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "unique_visitors": unique_visitors,
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "latest_sessions": latest_sessions_data,
            },
            status=status.HTTP_200_OK,
        )
