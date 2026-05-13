"""
Admin chatbot API views.

These APIs are used by the admin panel to review chatbot sessions,
messages, and chatbot statistics.
"""

from core.models import ChatSession
from django.db import models
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework import status

from core.models import ChatSession, ChatMessage, ContactMessage, ProjectRequest
from core.permissions import HasInternalAPIKey
from core.serializers import (
    AdminChatSessionListSerializer,
    AdminChatSessionDetailSerializer,
    AdminContactMessageSerializer,
    AdminProjectRequestSerializer
)
from django.db import connection
from django.conf import settings
from datetime import timedelta
from django.utils import timezone
from django.db.models import Count, Q
from django.db.models.functions import TruncDate

from django.db.models import Count, Min, Max


class AdminChatAnalyticsAPIView(APIView):
    """
    Admin API endpoint for chatbot analytics.

    Returns chart-ready analytics for:
    - Sessions by day
    - Messages by day
    - User messages by day
    - Assistant messages by day
    - Unique visitors by day
    - Total messages by role

    Protected by internal admin API key.
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        # Default: last 14 days
        try:
            range_days = int(request.query_params.get("days", 14))
        except ValueError:
            return Response(
                {"detail": "Invalid 'days' value. It must be a number."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Safety limit to avoid heavy queries
        if range_days < 1:
            range_days = 14

        if range_days > 90:
            range_days = 90

        end_date = timezone.now()
        start_date = end_date - timedelta(days=range_days)

        sessions_qs = ChatSession.objects.filter(created_at__gte=start_date)
        messages_qs = ChatMessage.objects.filter(created_at__gte=start_date)

        # Sessions grouped by date
        sessions_by_day = (
            sessions_qs
            .annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(
                sessions=Count("id"),
                unique_visitors=Count("visitor_id", distinct=True),
            )
            .order_by("day")
        )

        # Messages grouped by date
        messages_by_day = (
            messages_qs
            .annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(
                messages=Count("id"),
                user_messages=Count("id", filter=Q(role="user")),
                assistant_messages=Count("id", filter=Q(role="assistant")),
            )
            .order_by("day")
        )

        # Convert querysets to dictionaries by date
        sessions_map = {
            item["day"].isoformat(): {
                "sessions": item["sessions"],
                "unique_visitors": item["unique_visitors"],
            }
            for item in sessions_by_day
        }

        messages_map = {
            item["day"].isoformat(): {
                "messages": item["messages"],
                "user_messages": item["user_messages"],
                "assistant_messages": item["assistant_messages"],
            }
            for item in messages_by_day
        }

        # Merge sessions + messages into one chart-ready list
        analytics_by_day = []

        for i in range(range_days + 1):
            current_day = (start_date + timedelta(days=i)).date().isoformat()

            analytics_by_day.append({
                "date": current_day,
                "sessions": sessions_map.get(current_day, {}).get("sessions", 0),
                "messages": messages_map.get(current_day, {}).get("messages", 0),
                "user_messages": messages_map.get(current_day, {}).get("user_messages", 0),
                "assistant_messages": messages_map.get(current_day, {}).get("assistant_messages", 0),
                "unique_visitors": sessions_map.get(current_day, {}).get("unique_visitors", 0),
            })

        # Total messages by role
        messages_by_role = messages_qs.values(
            "role").annotate(count=Count("id"))

        role_counts = {
            "user": 0,
            "assistant": 0,
        }

        for item in messages_by_role:
            role_counts[item["role"]] = item["count"]

        return Response(
            {
                "range_days": range_days,
                "messages_by_day": analytics_by_day,
                "messages_by_role": role_counts,
            },
            status=status.HTTP_200_OK,
        )


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
    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

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


class AdminChatSessionDetailAPIView(APIView):
    """
    Admin API endpoint for reading one full chat session.

    Purpose:
    - Show complete visitor conversation
    - Review user questions and assistant responses
    - Inspect citations, confidence, metadata, and lead context
    - Support the admin conversation review page
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, session_id, *args, **kwargs):
        session = ChatSession.objects.filter(id=session_id).first()

        if not session:
            return Response(
                {"detail": "Chat session was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        messages_qs = (
            ChatMessage.objects
            .filter(session=session)
            .order_by("created_at")
        )

        messages = []

        for message in messages_qs:
            messages.append({
                "id": str(message.id),
                "role": message.role,
                "content": message.content,
                "citations": message.citations or [],
                "confidence_score": message.confidence_score,
                "metadata": message.metadata or {},
                "created_at": message.created_at,
            })

        user_messages_count = messages_qs.filter(role="user").count()
        assistant_messages_count = messages_qs.filter(role="assistant").count()

        low_confidence_count = messages_qs.filter(
            role="assistant",
            confidence_score__lt=0.7,
        ).count()

        fallback_count = 0
        missing_citations_count = 0

        for message in messages_qs.filter(role="assistant"):
            metadata = message.metadata or {}
            citations = message.citations or []

            if metadata.get("fallback_used") is True:
                fallback_count += 1

            if not citations:
                missing_citations_count += 1

        return Response(
            {
                "id": str(session.id),
                "visitor_id": session.visitor_id,
                "visitor_email": session.visitor_email,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "referrer": session.referrer,
                "is_active": session.is_active,


                "admin_status": session.admin_status,
                "admin_note": session.admin_note,
                "reviewed_at": session.reviewed_at,
                "closed_at": session.closed_at,


                "created_at": session.created_at,
                "updated_at": session.updated_at,

                "messages_count": messages_qs.count(),
                "user_messages_count": user_messages_count,
                "assistant_messages_count": assistant_messages_count,

                "quality_summary": {
                    "low_confidence_count": low_confidence_count,
                    "fallback_count": fallback_count,
                    "missing_citations_count": missing_citations_count,
                },

                "messages": messages,


            },
            status=status.HTTP_200_OK,
        )


class AdminChatSessionStatusUpdateAPIView(APIView):
    """
    Admin API endpoint for updating chat session review status.

    Used by the admin dashboard to mark sessions as:
    - open
    - reviewed
    - closed
    - archived

    PATCH /api/admin/chat/sessions/<session_id>/status/
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    allowed_statuses = ["open", "reviewed", "closed", "archived"]

    def patch(self, request, session_id, *args, **kwargs):
        session = ChatSession.objects.filter(id=session_id).first()

        if not session:
            return Response(
                {"detail": "Chat session was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        admin_status = request.data.get("admin_status")
        admin_note = request.data.get("admin_note", None)

        if admin_status not in self.allowed_statuses:
            return Response(
                {
                    "detail": "Invalid admin_status.",
                    "allowed_statuses": self.allowed_statuses,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        session.admin_status = admin_status

        if admin_note is not None:
            session.admin_note = str(admin_note).strip()

        now = timezone.now()

        if admin_status == "reviewed":
            session.reviewed_at = now

        if admin_status in ["closed", "archived"]:
            session.closed_at = now
            session.is_active = False

        if admin_status == "open":
            session.closed_at = None
            session.is_active = True

        session.save(
            update_fields=[
                "admin_status",
                "admin_note",
                "reviewed_at",
                "closed_at",
                "is_active",
                "updated_at",
            ]
        )

        return Response(
            {
                "success": True,
                "message": "Chat session status updated successfully.",
                "session": {
                    "id": str(session.id),
                    "admin_status": session.admin_status,
                    "admin_note": session.admin_note,
                    "reviewed_at": session.reviewed_at,
                    "closed_at": session.closed_at,
                    "is_active": session.is_active,
                    "updated_at": session.updated_at,
                },
            },
            status=status.HTTP_200_OK,
        )


class AdminChatStatsAPIView(APIView):
    """
    Show chatbot analytics summary for the admin dashboard.
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

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

        captured_emails = (
            ChatSession.objects
            .exclude(visitor_email__isnull=True)
            .exclude(visitor_email="")
            .values("visitor_email")
            .distinct()
            .count()
        )

        latest_sessions = (
            ChatSession.objects
            .all()
            .prefetch_related("messages")
            .order_by("-updated_at")[:5]
        )

        latest_sessions_data = AdminChatSessionListSerializer(
            latest_sessions,
            many=True,
        ).data

        return Response(
            {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "unique_visitors": unique_visitors,
                "captured_emails": captured_emails,
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "latest_sessions": latest_sessions_data,
            },
            status=status.HTTP_200_OK,
        )


class AdminLeadsAPIView(APIView):
    """
    Admin API endpoint for captured chatbot leads.

    A lead is any visitor/session where visitor_email is available.

    Returns:
    - visitor email
    - visitor id
    - number of sessions
    - number of messages
    - first seen date
    - last seen date
    - latest session id
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        search = request.query_params.get("search", "").strip()

        sessions_qs = ChatSession.objects.exclude(visitor_email__isnull=True).exclude(
            visitor_email__exact=""
        )

        if search:
            sessions_qs = sessions_qs.filter(visitor_email__icontains=search)

        grouped_leads = (
            sessions_qs
            .values("visitor_email", "visitor_id")
            .annotate(
                sessions_count=Count("id"),
                first_seen=Min("created_at"),
                last_seen=Max("updated_at"),
            )
            .order_by("-last_seen")
        )

        leads = []

        for lead in grouped_leads:
            visitor_email = lead["visitor_email"]
            visitor_id = lead["visitor_id"]

            lead_sessions = ChatSession.objects.filter(
                visitor_email=visitor_email,
                visitor_id=visitor_id,
            )

            latest_session = lead_sessions.order_by("-updated_at").first()

            messages_count = ChatMessage.objects.filter(
                session__in=lead_sessions
            ).count()

            leads.append({
                "visitor_email": visitor_email,
                "visitor_id": str(visitor_id) if visitor_id else None,
                "sessions_count": lead["sessions_count"],
                "messages_count": messages_count,
                "first_seen": lead["first_seen"],
                "last_seen": lead["last_seen"],
                "latest_session_id": str(latest_session.id) if latest_session else None,
            })

        return Response(
            {
                "total_leads": len(leads),
                "leads": leads,
            },
            status=status.HTTP_200_OK,
        )


class AdminContactMessagesAPIView(APIView):
    """
    Admin API endpoint for viewing saved Get in Touch form messages.

    Supports:
    - Listing all contact messages
    - Filtering by status
    - Searching by name, email, subject, or message
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        search = request.query_params.get("search", "").strip()
        status_filter = request.query_params.get("status", "").strip()

        messages_qs = ContactMessage.objects.all().order_by("-created_at")

        if status_filter:
            messages_qs = messages_qs.filter(status=status_filter)

        if search:
            messages_qs = messages_qs.filter(
                models.Q(name__icontains=search)
                | models.Q(email__icontains=search)
                | models.Q(subject__icontains=search)
                | models.Q(message__icontains=search)
            )

        serializer = AdminContactMessageSerializer(messages_qs, many=True)

        return Response(
            {
                "total_messages": messages_qs.count(),
                "messages": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class AdminContactMessagesAPIView(APIView):
    """
    Admin API endpoint for viewing saved Get in Touch form messages.

    Supports:
    - Listing all contact messages
    - Filtering by status
    - Searching by name, email, subject, or message
    """

    authentication_classes = []

    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        search = request.query_params.get("search", "").strip()
        status_filter = request.query_params.get("status", "").strip()

        messages_qs = ContactMessage.objects.all().order_by("-created_at")

        if status_filter:
            messages_qs = messages_qs.filter(status=status_filter)

        if search:
            messages_qs = messages_qs.filter(
                models.Q(name__icontains=search)
                | models.Q(email__icontains=search)
                | models.Q(subject__icontains=search)
                | models.Q(message__icontains=search)
            )

        serializer = AdminContactMessageSerializer(messages_qs, many=True)

        return Response(
            {
                "total_messages": messages_qs.count(),
                "messages": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class AdminContactMessageDetailAPIView(APIView):
    """
    Admin API endpoint for viewing or updating one contact message.

    Useful for changing status:
    - new
    - read
    - replied
    - archived
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, message_id, *args, **kwargs):
        contact_message = ContactMessage.objects.filter(id=message_id).first()

        if contact_message is None:
            return Response(
                {"detail": "Contact message not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = AdminContactMessageSerializer(contact_message)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, message_id, *args, **kwargs):
        contact_message = ContactMessage.objects.filter(id=message_id).first()

        if contact_message is None:
            return Response(
                {"detail": "Contact message not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        new_status = request.data.get("status")

        allowed_statuses = ["new", "read", "replied", "archived"]

        if new_status not in allowed_statuses:
            return Response(
                {
                    "detail": "Invalid status.",
                    "allowed_statuses": allowed_statuses,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        contact_message.status = new_status
        contact_message.save(update_fields=["status", "updated_at"])

        serializer = AdminContactMessageSerializer(contact_message)

        return Response(
            {
                "success": True,
                "message": "Contact message status updated successfully.",
                "contact_message": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class AdminProjectRequestsAPIView(APIView):
    """
    Admin API endpoint for viewing saved Start Project form requests.

    Supports:
    - Listing all project requests
    - Filtering by status
    - Filtering by project type
    - Searching by client name, email, project name, or description
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        search = request.query_params.get("search", "").strip()
        status_filter = request.query_params.get("status", "").strip()
        project_type = request.query_params.get("project_type", "").strip()

        requests_qs = ProjectRequest.objects.all().order_by("-created_at")

        if status_filter:
            requests_qs = requests_qs.filter(status=status_filter)

        if project_type:
            requests_qs = requests_qs.filter(project_type=project_type)

        if search:
            requests_qs = requests_qs.filter(
                models.Q(project_name__icontains=search)
                | models.Q(project_type__icontains=search)
                | models.Q(budget_range__icontains=search)
                | models.Q(timeline__icontains=search)
                | models.Q(project_description__icontains=search)
                | models.Q(your_name__icontains=search)
                | models.Q(your_email__icontains=search)
            )

        serializer = AdminProjectRequestSerializer(requests_qs, many=True)

        return Response(
            {
                "total_requests": requests_qs.count(),
                "project_requests": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class AdminProjectRequestDetailAPIView(APIView):
    """
    Admin API endpoint for viewing or updating one project request.

    Useful for changing status:
    - new
    - reviewed
    - contacted
    - accepted
    - rejected
    - archived
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, request_id, *args, **kwargs):
        project_request = ProjectRequest.objects.filter(id=request_id).first()

        if project_request is None:
            return Response(
                {"detail": "Project request not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = AdminProjectRequestSerializer(project_request)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, request_id, *args, **kwargs):
        project_request = ProjectRequest.objects.filter(id=request_id).first()

        if project_request is None:
            return Response(
                {"detail": "Project request not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        new_status = request.data.get("status")

        allowed_statuses = [
            "new",
            "reviewed",
            "contacted",
            "accepted",
            "rejected",
            "archived",
        ]

        if new_status not in allowed_statuses:
            return Response(
                {
                    "detail": "Invalid status.",
                    "allowed_statuses": allowed_statuses,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        project_request.status = new_status
        project_request.save(update_fields=["status", "updated_at"])

        serializer = AdminProjectRequestSerializer(project_request)

        return Response(
            {
                "success": True,
                "message": "Project request status updated successfully.",
                "project_request": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class AdminChatQualityIssuesAPIView(APIView):
    """
    Admin API endpoint for detecting chatbot quality issues.

    Detects assistant messages with:
    - Low confidence score
    - Safe fallback usage
    - Provider/model fallback usage
    - Empty citations for document/capability questions
    - Weak fallback phrases in the answer text

    Query params:
    - threshold: confidence threshold, default 0.5
    - limit: max issues returned, default 50, max 200
    - issue_type: optional filter
        low_confidence
        safe_fallback
        fallback_used
        missing_citations
        weak_answer_phrase
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    WEAK_ANSWER_PHRASES = [
        "i don’t have enough evidence",
        "i don't have enough evidence",
        "not enough evidence",
        "temporarily unable",
        "please try again",
        "i couldn’t find",
        "i couldn't find",
        "i do not have enough information",
        "i don't have enough information",
        "no relevant information",
    ]

    DOCUMENT_ROUTES = {
        "profile_docs_question",
        "capability_inference_question",
    }

    def get(self, request, *args, **kwargs):
        try:
            threshold = float(request.query_params.get("threshold", 0.5))
        except ValueError:
            return Response(
                {"detail": "Invalid threshold. It must be a number."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            limit = int(request.query_params.get("limit", 50))
        except ValueError:
            return Response(
                {"detail": "Invalid limit. It must be a number."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if limit < 1:
            limit = 50

        if limit > 200:
            limit = 200

        issue_type_filter = request.query_params.get("issue_type", "").strip()

        assistant_messages = (
            ChatMessage.objects
            .filter(role="assistant")
            .select_related("session")
            .order_by("-created_at")[:500]
        )

        issues = []

        for msg in assistant_messages:
            msg_issues = []

            metadata = msg.metadata or {}
            citations = msg.citations or []
            content = msg.content or ""
            content_lower = content.lower()
            confidence = msg.confidence_score

            question_route = metadata.get("question_route")
            answer_source = metadata.get("answer_source")
            mode = metadata.get("mode")

            if confidence is not None and confidence < threshold:
                msg_issues.append("low_confidence")

            if metadata.get("safe_fallback") is True:
                msg_issues.append("safe_fallback")

            if metadata.get("fallback_used") is True:
                msg_issues.append("fallback_used")

            if question_route in self.DOCUMENT_ROUTES and not citations:
                msg_issues.append("missing_citations")

            if any(phrase in content_lower for phrase in self.WEAK_ANSWER_PHRASES):
                msg_issues.append("weak_answer_phrase")

            if not msg_issues:
                continue

            if issue_type_filter and issue_type_filter not in msg_issues:
                continue

            previous_user_message = (
                ChatMessage.objects
                .filter(
                    session=msg.session,
                    role="user",
                    created_at__lt=msg.created_at,
                )
                .order_by("-created_at")
                .first()
            )

            issues.append({
                "message_id": str(msg.id),
                "session_id": str(msg.session_id),
                "visitor_id": msg.session.visitor_id,
                "visitor_email": msg.session.visitor_email,
                "issues": msg_issues,
                "confidence_score": confidence,
                "question_route": question_route,
                "answer_source": answer_source,
                "mode": mode,
                "model_used": metadata.get("model_used"),
                "provider_used": metadata.get("provider_used"),
                "fallback_used": metadata.get("fallback_used"),
                "safe_fallback": metadata.get("safe_fallback"),
                "retrieval_query": metadata.get("retrieval_query"),
                "rewrite_notes": metadata.get("rewrite_notes"),
                "citations_count": len(citations),
                "user_message": previous_user_message.content if previous_user_message else None,
                "assistant_answer": content,
                "created_at": msg.created_at,
            })

            if len(issues) >= limit:
                break

        issue_counts = {
            "low_confidence": 0,
            "safe_fallback": 0,
            "fallback_used": 0,
            "missing_citations": 0,
            "weak_answer_phrase": 0,
        }

        for issue in issues:
            for issue_name in issue["issues"]:
                issue_counts[issue_name] = issue_counts.get(issue_name, 0) + 1

        return Response(
            {
                "threshold": threshold,
                "limit": limit,
                "total_issues": len(issues),
                "issue_counts": issue_counts,
                "issues": issues,
            },
            status=status.HTTP_200_OK,
        )


class AdminDashboardSummaryAPIView(APIView):
    """
    Admin dashboard summary API.

    Returns compact metrics for the dashboard overview cards:
    - Chat sessions/messages
    - Leads
    - Contact messages
    - Project requests
    - Chat quality issues summary
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        # Chat metrics
        total_sessions = ChatSession.objects.count()
        active_sessions = ChatSession.objects.filter(is_active=True).count()
        total_messages = ChatMessage.objects.count()
        user_messages = ChatMessage.objects.filter(role="user").count()
        assistant_messages = ChatMessage.objects.filter(
            role="assistant").count()

        # Leads are chatbot sessions with captured visitor email
        total_leads = (
            ChatSession.objects
            .exclude(visitor_email__isnull=True)
            .exclude(visitor_email__exact="")
            .values("visitor_email", "visitor_id")
            .distinct()
            .count()
        )

        # Contact messages
        contact_total = ContactMessage.objects.count()
        contact_new = ContactMessage.objects.filter(status="new").count()
        contact_read = ContactMessage.objects.filter(status="read").count()
        contact_replied = ContactMessage.objects.filter(
            status="replied").count()
        contact_archived = ContactMessage.objects.filter(
            status="archived").count()

        # Project requests
        project_total = ProjectRequest.objects.count()
        project_new = ProjectRequest.objects.filter(status="new").count()
        project_reviewed = ProjectRequest.objects.filter(
            status="reviewed").count()
        project_contacted = ProjectRequest.objects.filter(
            status="contacted").count()
        project_accepted = ProjectRequest.objects.filter(
            status="accepted").count()
        project_rejected = ProjectRequest.objects.filter(
            status="rejected").count()
        project_archived = ProjectRequest.objects.filter(
            status="archived").count()

        # Quality summary
        assistant_qs = ChatMessage.objects.filter(role="assistant")

        low_confidence_count = assistant_qs.filter(
            confidence_score__isnull=False,
            confidence_score__lt=0.5,
        ).count()

        fallback_used_count = assistant_qs.filter(
            metadata__fallback_used=True,
        ).count()

        safe_fallback_count = assistant_qs.filter(
            metadata__safe_fallback=True,
        ).count()

        missing_citations_count = 0

        for msg in assistant_qs:
            metadata = msg.metadata or {}
            citations = msg.citations or []
            question_route = metadata.get("question_route")

            if question_route in ["profile_docs_question", "capability_inference_question"] and not citations:
                missing_citations_count += 1

        quality_issues_total = (
            low_confidence_count
            + fallback_used_count
            + safe_fallback_count
            + missing_citations_count
        )

        latest_contact_message = ContactMessage.objects.order_by(
            "-created_at").first()
        latest_project_request = ProjectRequest.objects.order_by(
            "-created_at").first()
        latest_chat_session = ChatSession.objects.order_by(
            "-updated_at").first()

        return Response(
            {
                "chat": {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "total_messages": total_messages,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages,
                    "latest_session_id": str(latest_chat_session.id) if latest_chat_session else None,
                },
                "leads": {
                    "total_leads": total_leads,
                },
                "contact_messages": {
                    "total": contact_total,
                    "new": contact_new,
                    "read": contact_read,
                    "replied": contact_replied,
                    "archived": contact_archived,
                    "latest_contact_message_id": str(latest_contact_message.id) if latest_contact_message else None,
                },
                "project_requests": {
                    "total": project_total,
                    "new": project_new,
                    "reviewed": project_reviewed,
                    "contacted": project_contacted,
                    "accepted": project_accepted,
                    "rejected": project_rejected,
                    "archived": project_archived,
                    "latest_project_request_id": str(latest_project_request.id) if latest_project_request else None,
                },
                "quality": {
                    "total_issues": quality_issues_total,
                    "low_confidence": low_confidence_count,
                    "fallback_used": fallback_used_count,
                    "safe_fallback": safe_fallback_count,
                    "missing_citations": missing_citations_count,
                },
            },
            status=status.HTTP_200_OK,
        )


class AdminNotificationBadgesAPIView(APIView):
    """
    Admin notification badges API.

    Returns small counts for sidebar/header badges:
    - New contact messages
    - New project requests
    - Quality issues
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        assistant_qs = ChatMessage.objects.filter(role="assistant")

        low_confidence_count = assistant_qs.filter(
            confidence_score__isnull=False,
            confidence_score__lt=0.5,
        ).count()

        fallback_used_count = assistant_qs.filter(
            metadata__fallback_used=True,
        ).count()

        safe_fallback_count = assistant_qs.filter(
            metadata__safe_fallback=True,
        ).count()

        missing_citations_count = 0

        for msg in assistant_qs:
            metadata = msg.metadata or {}
            citations = msg.citations or []
            question_route = metadata.get("question_route")

            if question_route in ["profile_docs_question", "capability_inference_question"] and not citations:
                missing_citations_count += 1

        quality_issues_count = (
            low_confidence_count
            + fallback_used_count
            + safe_fallback_count
            + missing_citations_count
        )

        return Response(
            {
                "new_contact_messages": ContactMessage.objects.filter(status="new").count(),
                "new_project_requests": ProjectRequest.objects.filter(status="new").count(),
                "quality_issues": quality_issues_count,
                "low_confidence_answers": low_confidence_count,
                "fallback_used": fallback_used_count,
                "safe_fallback": safe_fallback_count,
                "missing_citations": missing_citations_count,
            },
            status=status.HTTP_200_OK,
        )


class AdminChatSessionExportAPIView(APIView):
    """
    Admin API endpoint for exporting one chat session.

    Returns a JSON export of:
    - Session metadata
    - All messages
    - Citations
    - Confidence scores
    - Message metadata
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, session_id, *args, **kwargs):
        session = ChatSession.objects.filter(id=session_id).first()

        if session is None:
            return Response(
                {"detail": "Chat session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        messages = (
            ChatMessage.objects
            .filter(session=session)
            .order_by("created_at")
        )

        exported_messages = []

        for msg in messages:
            exported_messages.append({
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations or [],
                "confidence_score": msg.confidence_score,
                "metadata": msg.metadata or {},
                "created_at": msg.created_at,
            })

        return Response(
            {
                "export_type": "chat_session",
                "session": {
                    "id": str(session.id),
                    "visitor_id": session.visitor_id,
                    "visitor_email": session.visitor_email,
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "referrer": session.referrer,
                    "is_active": session.is_active,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                },
                "messages_count": messages.count(),
                "messages": exported_messages,
            },
            status=status.HTTP_200_OK,
        )


class AdminSystemHealthAPIView(APIView):
    """
    Admin API endpoint for checking backend and database health.

    Useful for the admin dashboard system status widget.
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        database_status = "connected"

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception:
            database_status = "error"

        overall_status = "ok" if database_status == "connected" else "degraded"

        return Response(
            {
                "status": overall_status,
                "database": database_status,
                "server_time": timezone.now(),
                "debug": settings.DEBUG,
                "environment": getattr(settings, "ENVIRONMENT", "development"),
                "counts": {
                    "chat_sessions": ChatSession.objects.count(),
                    "chat_messages": ChatMessage.objects.count(),
                    "contact_messages": ContactMessage.objects.count(),
                    "project_requests": ProjectRequest.objects.count(),
                },
            },
            status=status.HTTP_200_OK,
        )


class AdminRecentActivityAPIView(APIView):
    """
    Admin API endpoint for showing recent activity across the portfolio system.

    Combines:
    - Latest chatbot messages
    - Latest contact form messages
    - Latest project requests

    Query params:
    - limit: max activities returned, default 20, max 100
    """
    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        try:
            limit = int(request.query_params.get("limit", 20))
        except ValueError:
            return Response(
                {"detail": "Invalid limit. It must be a number."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if limit < 1:
            limit = 20

        if limit > 100:
            limit = 100

        activities = []

        # Recent user chat messages only, because these represent visitor activity
        recent_user_messages = (
            ChatMessage.objects
            .filter(role="user")
            .select_related("session")
            .order_by("-created_at")[:limit]
        )

        for msg in recent_user_messages:
            preview = (msg.content or "").strip()
            if len(preview) > 120:
                preview = preview[:120] + "..."

            activities.append({
                "type": "chat_message",
                "label": "New chatbot message",
                "description": f"Visitor asked: {preview}",
                "reference_id": str(msg.id),
                "session_id": str(msg.session_id),
                "visitor_email": msg.session.visitor_email,
                "visitor_id": msg.session.visitor_id,
                "created_at": msg.created_at,
            })

        # Recent contact form messages
        recent_contact_messages = (
            ContactMessage.objects
            .all()
            .order_by("-created_at")[:limit]
        )

        for item in recent_contact_messages:
            activities.append({
                "type": "contact_message",
                "label": "New contact message",
                "description": f"Message from {item.name} ({item.email})",
                "reference_id": str(item.id),
                "status": item.status,
                "created_at": item.created_at,
            })

        # Recent project requests
        recent_project_requests = (
            ProjectRequest.objects
            .all()
            .order_by("-created_at")[:limit]
        )

        for item in recent_project_requests:
            activities.append({
                "type": "project_request",
                "label": "New project request",
                "description": f"{item.project_name} from {item.your_name} ({item.your_email})",
                "reference_id": str(item.id),
                "status": item.status,
                "project_type": item.project_type,
                "created_at": item.created_at,
            })

        # Sort all activity together by date
        activities = sorted(
            activities,
            key=lambda x: x["created_at"],
            reverse=True,
        )[:limit]

        return Response(
            {
                "total_returned": len(activities),
                "activities": activities,
            },
            status=status.HTTP_200_OK,
        )


class AdminChatMessagesAPIView(APIView):
    """
    Admin API endpoint for listing chatbot messages directly.

    This endpoint is useful for dashboard drill-down views, such as:
    - Viewing all user messages
    - Viewing all assistant messages
    - Searching message content
    - Checking which visitor/session a message belongs to

    Query params:
    - role: optional filter. Accepted values: user, assistant
    - search: optional keyword search across message content, visitor email, and visitor id
    - limit: optional maximum records returned. Default 100, max 500

    Example:
    GET /api/admin/chat/messages/?role=user
    GET /api/admin/chat/messages/?role=assistant
    GET /api/admin/chat/messages/?role=user&search=project
    """

    authentication_classes = []
    permission_classes = [HasInternalAPIKey]
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        role = request.query_params.get("role", "").strip()
        search = request.query_params.get("search", "").strip()

        try:
            limit = int(request.query_params.get("limit", 100))
        except ValueError:
            return Response(
                {"detail": "Invalid limit. It must be a number."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if limit < 1:
            limit = 100

        if limit > 500:
            limit = 500

        messages_qs = (
            ChatMessage.objects
            .select_related("session")
            .all()
            .order_by("-created_at")
        )

        if role:
            if role not in ["user", "assistant"]:
                return Response(
                    {
                        "detail": "Invalid role.",
                        "allowed_roles": ["user", "assistant"],
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            messages_qs = messages_qs.filter(role=role)

        if search:
            messages_qs = messages_qs.filter(
                models.Q(content__icontains=search)
                | models.Q(session__visitor_email__icontains=search)
                | models.Q(session__visitor_id__icontains=search)
            )

        total_messages = messages_qs.count()
        messages_qs = messages_qs[:limit]

        messages = []

        for message in messages_qs:
            related_user_message = None

            # If this is an assistant answer, find the closest previous user message
            # in the same chat session. This helps the admin understand which user
            # question led to this assistant response.
            if message.role == "assistant":
                previous_user_message = (
                    ChatMessage.objects
                    .filter(
                        session_id=message.session_id,
                        role="user",
                        created_at__lt=message.created_at,
                    )
                    .order_by("-created_at")
                    .first()
                )

                if previous_user_message:
                    related_user_message = {
                        "id": str(previous_user_message.id),
                        "content": previous_user_message.content,
                        "created_at": previous_user_message.created_at,
                    }

            messages.append({
                "id": str(message.id),
                "session_id": str(message.session_id),
                "role": message.role,
                "content": message.content,
                "citations": message.citations or [],
                "confidence_score": message.confidence_score,
                "metadata": message.metadata or {},
                "visitor_email": message.session.visitor_email,
                "visitor_id": message.session.visitor_id,
                "created_at": message.created_at,
                "related_user_message": related_user_message,
            })

        return Response(
            {
                "total_messages": total_messages,
                "returned_messages": len(messages),
                "role": role or "all",
                "search": search,
                "limit": limit,
                "messages": messages,
            },
            status=status.HTTP_200_OK,
        )
